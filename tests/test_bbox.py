from core.geometry import BoundingBox2D

bbox1 = BoundingBox2D.from_xywh(
    100,
    100,
    80,
    60,
)

bbox2 = BoundingBox2D.from_xywh(
    120,
    120,
    70,
    70,
)

print("=" * 60)
print("Advanced Metrics")
print("=" * 60)

print("IoU  :", bbox1.iou(bbox2))
print("IoA  :", bbox1.ioa(bbox2))
print("GIoU :", bbox1.giou(bbox2))
print("DIoU :", bbox1.diou(bbox2))
print("CIoU :", bbox1.ciou(bbox2))