from tracking.association import CostMatrixBuilder
from tracking.metrics import EuclideanMetric

builder = CostMatrixBuilder(
    metric=EuclideanMetric()
)

print("=" * 60)
print(builder)
print("=" * 60)

print(builder.metric_name)