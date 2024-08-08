import numpy as np
from enum import Enum
import json
import base64
from PIL import Image
import io
import cv2 as cv


class AnnotationLabel(Enum):
    POD = 0
    SEED = 1
    INTERIOR = 2
    ENDOSPERM = 3
    INFESTATION = 4
    VOID = 5
    EMBRYO = 6


class ModelAnnotation:
    def __init__(
        self,
        bbox: list[float],
        mask: np.ndarray,
        label: AnnotationLabel,
        confidence: float,
        area: float,
        mean_intensity: float,
        seed_id: int,
    ) -> None:
        
        self.bbox = bbox
        self.mask = mask
        self.label = label
        self.confidence = confidence
        self.area = area
        self.mean_intensity = mean_intensity
        self.seed_id = None
        self.hull = self.get_hull()

    def get_hull(self) -> np.ndarray:
        if self.mask.any():
            contours, _ = cv.findContours(
                self.mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
            )
            hull = cv.convexHull(np.vstack(contours))
            hull_mask = cv.drawContours(np.zeros_like(self.mask), [hull], -1, 255, -1)
            return hull_mask
        else:
            return np.zeros_like(self.mask)
    
    def overlap(self, other: 'ModelAnnotation') -> float:
        intersection = cv.bitwise_and(self.hull, other.hull).sum()
        union = cv.bitwise_or(self.hull, other.hull).sum()
        if union > 0:
            return intersection / union
        else:
            return 0
        
    def bbox_overlap(self, other: "ModelAnnotation") -> bool:
        self_min_x = self.bbox[0]
        self_min_y = self.bbox[1]
        self_max_x = self_min_x + self.bbox[2]
        self_max_y = self_min_y + self.bbox[3]

        other_min_x = other.bbox[0]
        other_min_y = other.bbox[1]
        other_max_x = other_min_x + other.bbox[2]
        other_max_y = other_min_y + other.bbox[3]

        return not(self_max_x < other_min_x 
                   or self_min_x > other_max_x 
                   or self_min_y > other_max_y 
                   or self_max_y < other_min_y)

    def to_json(self) -> dict:
        mask = add_transparency(self.mask)
        mask_str = get_base64_encoding(mask)
        return {
            "bbox": self.bbox,
            "mask": mask_str,
            "label": self.label.value,
            "confidence": self.confidence,
            "area": self.area,
            "mean_intensity": self.mean_intensity,
            "seed_id": self.seed_id,
        }


class ModelResponse:
    def __init__(self, annotations: list[ModelAnnotation], width: int, height: int) -> None:
        self.annotations = annotations
        self.width = width
        self.height = height
        self.populate_seed_indices()

    def to_json(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "annotations": [a.to_json() for a in self.annotations],
            "composite_masks": {k: get_base64_encoding(add_transparency(v)) 
                                for k,v in self.get_composite_masks().items()},
            "seed_indices": self.flatten_seed_indices(),
        }

    def get_composite_masks(self) -> dict[int, str]:
        distinct_labels = {annotation.label.value for annotation in self.annotations}
        composite_masks = {}
        for label in distinct_labels:
            composite_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for annotation in self.annotations:
                if annotation.label.value != label:
                    continue
                composite_mask = cv.bitwise_or(composite_mask, annotation.mask)
            composite_masks[label] = composite_mask
        return composite_masks

    def flatten_seed_indices(self) -> list[list[int]]:
        seed_dict = {}
        for i, annotation in enumerate(self.annotations):
            if annotation.seed_id not in seed_dict:
                seed_dict[annotation.seed_id] = []
            seed_dict[annotation.seed_id].append(i)

        return [seed_dict[k] for k in seed_dict.keys() if k is not None] # TODO: return orphaned seed indices

    def populate_seed_indices(self, threshold = 0.05):
        seeds = [x for x in self.annotations if x.label == AnnotationLabel.SEED]
        other = [x for x in self.annotations if x.label != AnnotationLabel.SEED]

        for i, seed in enumerate(seeds):
            seed.seed_id = i
        
        for annotation in other:
            seed_id = None
            max_overlap = 0
            for seed in seeds:
                if not annotation.bbox_overlap(seed):
                    continue
                overlap = annotation.overlap(seed)
                if overlap > max_overlap:
                    seed_id = seed.seed_id
                    max_overlap = overlap
            if max_overlap > threshold:
                annotation.seed_id = seed_id
            else:
                annotation.seed_id = None


def get_base64_encoding(mask: np.ndarray) -> str:
    mask_image = Image.fromarray(mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="png")
    mask_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return mask_str


def add_transparency(mask) -> np.ndarray:
    transparency = np.zeros_like(mask)
    transparency[mask > 0] = 128
    new_image = np.dstack((mask, transparency))
    return new_image


if __name__ == "__main__":
    """ annotations = [
        ModelAnnotation(
            bbox=[0, 0, 5, 5],
            mask=np.full((20, 20), 123, dtype=np.uint8),
            label=AnnotationLabel.SEED,
            confidence=0.9,
            area=100,
            mean_intensity=128,
            seed_id=-1,
        ),
        ModelAnnotation(
            bbox=[4, 4, 5, 5],
            mask=np.full((20, 20), 123, dtype=np.uint8),
            label=AnnotationLabel.SEED,
            confidence=0.9,
            area=100,
            mean_intensity=128,
            seed_id=-1,
        ),
        ModelAnnotation(
            bbox=[1, 4, 1, 1],
            mask=np.full((20, 20), 123, dtype=np.uint8),
            label=AnnotationLabel.VOID,
            confidence=0.9,
            area=100,
            mean_intensity=128,
            seed_id=-1,
        ),
        ModelAnnotation(
            bbox=[4, 4, 0.5, 0.5],
            mask=np.full((20, 20), 123, dtype=np.uint8),
            label=AnnotationLabel.INFESTATION,
            confidence=0.9,
            area=100,
            mean_intensity=128,
            seed_id=-1,
        ),
    ] """
    mask1 = np.zeros((10,10), dtype=np.uint8)
    mask2 = np.zeros((10,10), dtype=np.uint8)
    mask1[1,1] = 255
    mask1[4,1] = 255
    mask1[1,4] = 255
    mask1[4,4] = 255
    mask2[3:6, 3:6] = 255
    annotations = [
        ModelAnnotation(
            bbox=[0,0,10,10],
            mask=mask1,
            label=AnnotationLabel.SEED,
            confidence=0.95,
            area = 1.0,
            mean_intensity=255,
            seed_id=-1
        ),
        ModelAnnotation(
            bbox=[0,0,10,10],
            mask=mask2,
            label=AnnotationLabel.SEED,
            confidence=0.95,
            area = 1.0,
            mean_intensity=255,
            seed_id=-1
        )
    ]
    response = ModelResponse(width=20, height=20, annotations=annotations)
    print(json.dumps(response.to_json(), indent=4))
