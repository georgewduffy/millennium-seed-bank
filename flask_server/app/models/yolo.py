from .base_model import Model
from ultralytics import YOLO as UltralyticsYOLO
import os
from PIL import Image
import numpy as np
import torch
from ..routes.responses import ModelAnnotation, ModelResponse, AnnotationLabel
import torch.nn.functional as F


THRESHOLD = 0.0
annotation_map = {
    0:3, # endosperm
    1:4, # infestation
    2:2, # interior
    3:1, # seed
    4:5, # void
}

class YOLO(Model):
    def __init__(self, weights_path="app/models/final_model_weights/yolo.pt"):
        super(YOLO, self).__init__()
        self.model = self.build_model(weights_path)

    def build_model(self, weights_path):
        if weights_path is None:
            raise FileNotFoundError(f"Could not find yolo model weights at {weights_path}")

        return UltralyticsYOLO(weights_path)

    def predict(self, input):
        image = Image.fromarray(input)
        device = "cpu"
        result = list(self.model(image, stream=True))[0]

        # get indices of scores above threshold
        kept_indices = np.where(result.boxes.conf.to(device) > THRESHOLD)[0]
        # for mask in result.masks.data:
        #     print(mask.shape)

        # masks = [F.interpolate(mask, size=(image.height, image.width), mode='nearest') for mask in result.masks.data ]
        # print(result.masks.data.shape)
        # masks = F.interpolate(result.masks.data, size=(result.masks.data.shape[0],image.height, image.width))
        masks = F.interpolate(result.masks.data.unsqueeze(1), size=(image.height, image.width), mode='nearest').squeeze(1)

        annotations = [
            ModelAnnotation(
                bbox=self.get_response_bbox(result.boxes.xyxy[idx].to(device).tolist()),
                mask=np.floor(masks[idx].to(device).numpy()*255).astype(np.uint8),
                label=AnnotationLabel( annotation_map[int(result.boxes.cls[idx].to(torch.int).item())] ),
                mean_intensity = self.calculate_average_intensity(masks[idx].to(device).numpy(), np.array(image)),
                area = self.calculate_area(masks[idx].to(device).numpy()),
                confidence = result.boxes.conf[idx].to(device).item(),
                seed_id=-1,
            )
            for idx in kept_indices
        ]




        # Return a ModelResponse object
        return ModelResponse(annotations=annotations, width=image.width, height=image.height)

    def calculate_average_intensity(self, mask, image):
        mask = mask.astype(bool)
        
        if np.all(~mask):
            return 0.0

        return float(np.mean(image[mask]))
    
    def calculate_area(self, mask):
        mask = mask.astype(bool)
        return float(np.sum(mask))
    
    def get_response_bbox(self, bbox: list[float]) -> list[float]:
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    

if __name__ == '__main__':
    print("----- Testing yolo module -----\n\n")
    yolo = YOLO()
    print("built model")
    test_image_name = '9c9128a7-0699125_XRAY_1.jpg'
    full_path = None

    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        if test_image_name in filenames:
            full_path = os.path.join(dirpath, test_image_name)
            
    if full_path is None:
        raise FileNotFoundError(f"Could not find test image {test_image_name}")
    
    print(f"Testing with image {full_path}")

    numpy_image = np.array(Image.open(full_path))
    result = yolo.predict(numpy_image)
    print(result.to_json())