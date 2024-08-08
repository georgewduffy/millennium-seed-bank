import math


class BoundingBox:
    def __init__(self, x: float, y: float, width: float, height: float, box_id: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.box_id = box_id

    def contains(self, other: "BoundingBox") -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.x + other.width <= self.x + self.width
            and other.y + other.height <= self.y + self.height
        )

    def is_contained(self, other: "BoundingBox") -> bool:
        return other.contains(self)

    def midpoint(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def midpoint_distance(self, other: "BoundingBox") -> float:
        self_mid = self.midpoint()
        other_mid = other.midpoint()
        return math.sqrt(
            (self_mid[0] - other_mid[0]) ** 2 + (self_mid[1] - other_mid[1]) ** 2
        )
    
    def __repr__(self) -> str:
        return f"{(self.x, self.y, self.width, self.height)}"


class BoundingBoxCollection:
    def __init__(self, root: BoundingBox) -> None:
        self.root = root
        self.boxes = [root]

    def add(self, box: BoundingBox) -> None:
        self.boxes.append(box)

    def mean_distance(self, other: BoundingBox) -> float:
        dist = 0
        count = 0
        for box in self.boxes:
            dist += other.midpoint_distance(box)
            count += 1
        return dist / count

    def contains(self, box: BoundingBox) -> bool:
        return self.root.contains(box)
    
    def __repr__(self) -> str:
        return f"{self.root} -> {self.boxes[1:]}"


def get_bbox_hierarchy(bboxs: list[BoundingBox]) -> list[BoundingBoxCollection]:
    # Root boxes
    root_boxes = []
    non_root_boxes = []
    for box in bboxs:
        root = True
        for other_box in bboxs:
            if other_box is box:
                continue
            if other_box.contains(box):
                root = False
                break
        if root:
            bbox_collection = BoundingBoxCollection(box)
            root_boxes.append(bbox_collection)
        else:
            non_root_boxes.append(box)

    # Unambiguous boxes
    ambiguous_boxes = []
    for box in non_root_boxes:
        parents = []
        for root in root_boxes:
            if root.contains(box):
                parents.append(root)
        if len(parents) == 1:
            parents[0].add(box)
        else:
            ambiguous_boxes.append(box)

    # Ambiguous boxes
    for box in ambiguous_boxes:
        min_dist = math.inf
        nearest = None
        for root in root_boxes:
            dist = root.mean_distance(box)
            if dist < min_dist:
                min_dist = dist
                nearest = root
        nearest.add(box)
    
    return root_boxes

if __name__ == "__main__":
    bboxs = [
        BoundingBox(0,0,5,5),
        BoundingBox(4,4,5,5),
        BoundingBox(1,4,1,1),
        BoundingBox(4,4,0.5,0.5),
    ]
    print(get_bbox_hierarchy(bboxs))
