import numpy as np
from enum import Enum

class AnnotationLabel(Enum):
    POD = 0
    SEED = 1
    INTERIOR = 2
    ENDOSPERM = 3
    INFESTATION = 4
    VOID = 5

def groupMasksBySeed(annotations):
    seed_boxes = [(i, ann.bbox) for i, ann in enumerate(annotations) if ann.label == AnnotationLabel.SEED]
    groups = {seed_index: [] for seed_index, _ in seed_boxes}

    for i, ann in enumerate(annotations):
        if ann.label != AnnotationLabel.SEED:
            mask = ann.mask
            mask_indices = np.where(mask == 255)
            mask_coords = np.vstack((mask_indices[1], mask_indices[0])).T

            for seed_index, bbox in seed_boxes:
                if mask_in_bbox(mask_coords, bbox):
                    groups[seed_index].append(i)

    return [(seed, masks) for seed, masks in groups.items()]

def mask_in_bbox(mask_coords, bbox):
    x, y, w, h = bbox
    return np.all((mask_coords[:, 0] >= x) & (mask_coords[:, 0] < x + w) &
                  (mask_coords[:, 1] >= y) & (mask_coords[:, 1] < y + h))