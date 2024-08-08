from .base_model import Model
import torch
import numpy as np
from flask_server.app.routes.responses import AnnotationLabel, ModelAnnotation, ModelResponse
from torchvision.transforms import v2
from src.mask_rcnn_training.training_utils.ring_mask_converter import process
from src.mask_rcnn_training.training_utils.utils import NestedTensorHandler, convert_labels#, print_structure
from src.mask_rcnn_training.training_utils.model_builder import get_model
import json

class RCNN(Model):
    def __init__(self, configs_path='app/models/rcnn_inference_configs/inference_configs.json'):
        super(RCNN, self).__init__()
        self.configs_path = configs_path
        self.inference_configs, self.training_configs = self.load_configs()
        self.hyper_params = self.training_configs['hyper_params']
        self.device = torch.device('cuda:0' if self.inference_configs['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.label_order, self.id2label, self.label2id, self.name2id, self.id2name, self.name2label, self.label2name = self.get_conversions()
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x / 255.0),
        ])
        self.name2annotation = dict([
            ("Pod", AnnotationLabel.POD),
            ("Seed", AnnotationLabel.SEED),
            ("Interior", AnnotationLabel.INTERIOR),
            ("Endosperm", AnnotationLabel.ENDOSPERM),
            ("Embryo", AnnotationLabel.EMBRYO),
            ("Void", AnnotationLabel.VOID),
            ("Infestation", AnnotationLabel.INFESTATION),        
        ])

    def get_conversions(self):
        label_order = torch.tensor(self.training_configs['label_order'])
        id2label = dict([(int(key), int(value)) for key, value in self.training_configs['id2label'].items()])
        label2id = dict([(int(key), int(value)) for key, value in self.training_configs['label2id'].items()])
        name2id = dict([(str(key), int(value)) for key, value in self.training_configs['name2id'].items()])
        id2name = dict([(int(key), str(value)) for key, value in self.training_configs['id2name'].items()])
        name2label = dict([(str(key), int(value)) for key, value in self.training_configs['name2label'].items()])
        label2name = dict([(int(key), str(value)) for key, value in self.training_configs['label2name'].items()])
        return label_order, id2label, label2id, name2id, id2name, name2label, label2name

    def preprocess(self, images, targets):
        targets = convert_labels(targets, self.id2label)
        if self.hyper_params['pre-process']:
            images, targets = process(images, targets, self.label_order)
        images = NestedTensorHandler.get_structure_on_device(images, self.device)
        targets = NestedTensorHandler.get_structure_on_device(targets, self.device)
        return images, targets

    def postprocess(self, images, preds):
        if self.hyper_params['post-process']:
            return process(images, preds, self.label_order)
        else:
            return images, preds
        
    def load_configs(self):
        with open(self.configs_path, 'r') as file:
            inference_configs = json.load(file)
        with open(inference_configs['fixed_inference_configs_path'], 'r') as file:
            training_configs = json.load(file)
        return inference_configs, training_configs
    
    def build_model(self):
        model = get_model(
            self.training_configs['num_classes'] + 1,
            max_detections=self.hyper_params['max_detections'],
            model_path=self.training_configs['model_path'],
        ).to(self.device)
        model.eval()
        return model

    def convert_bboxes(self, bboxes: torch.Tensor) -> list[list]:
        boxes = []
        for i in range(len(bboxes)):
            x, y, x2, y2 = bboxes[i].tolist()
            boxes.append([x, y, x2-x, y2-y])
        return boxes

    def deconvert(self, labels):
        return torch.tensor([self.label2id[int(label)] for label in labels])

    def binarise(self, pred):
        pred['masks'] = (pred['masks'] > self.hyper_params['binary_threshold']).to(self.device)
        return pred

    def get_annotations(self, numpy_image: np.ndarray, pred: dict[str, torch.tensor]) -> ModelAnnotation:
        annotations = []
        for i in range(len(pred['labels'])):
            bbox = pred['boxes'][i]
            mask = pred['masks'][i].numpy().squeeze().astype(np.uint8)
            name = self.label2name[int(pred['labels'][i])]
            label = self.name2annotation[name]
            area = float(mask.sum())
            mean_intensity = float((numpy_image*mask).sum()/(area+1e-5))
            confidence = float(pred['scores'][i])
            seed_id = None
            annotations.append(ModelAnnotation(bbox=bbox, mask=mask*255, label=label, area=area, mean_intensity=mean_intensity, confidence=confidence, seed_id=seed_id))
        return annotations        

    def predict_internal(self, torch_image: torch.tensor):
        torch_image = self.transforms(torch_image)
        with torch.no_grad():
            pred = self.model([torch_image])[0]
        #print(torch.unique(pred['masks']))
        pred = self.binarise(pred)
        pred = self.postprocess([torch_image], [pred])[1][0]
        #print(torch.unique(pred['masks']))
        return NestedTensorHandler.get_structure_on_device(pred, 'cpu')
                
    def predict(self, numpy_image: np.ndarray):
        height, width = numpy_image.shape
        torch_image = torch.tensor(numpy_image, dtype=torch.float).view(1, height, width)
        pred = self.predict_internal(torch_image)
        pred['boxes'] = self.convert_bboxes(pred['boxes'])
        annotations = self.get_annotations(numpy_image, pred)
        return ModelResponse(annotations=annotations, height=height, width=width)
    