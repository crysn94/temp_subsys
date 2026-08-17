from core.geometry import BoundingBox3D

a = BoundingBox3D.from_center(
    0,
    0,
    0,
    10,
    10,
    10,
)

b = BoundingBox3D.from_center(
    4,
    4,
    4,
    10,
    10,
    10,
)

print("=" * 60)
print("BoundingBox3D")
print("=" * 60)

print("Volume       :", a.volume)
print("Center       :", a.center)
print("Intersection :", a.intersection(b))
print("Union        :", a.union(b))
print("IoU          :", a.iou(b))
print("Valid        :", a.is_valid)
print("Translated   :", a.translate(5, 0, 0))