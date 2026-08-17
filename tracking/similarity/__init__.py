"""
========================================================================
Tracking Similarity Metrics
========================================================================

Canonical association metrics used by the tracking subsystem.

Lower cost = better association.

========================================================================
"""

from tracking.similarity.base_metric import BaseMetric
from tracking.similarity.euclidean import EuclideanMetric
from tracking.similarity.mahalanobis import MahalanobisMetric
from tracking.similarity.iou import IoUMetric
from tracking.similarity.giou import GIoUMetric
from tracking.similarity.motion import MotionMetric
from tracking.similarity.hybrid import HybridMetric


__all__ = [
    "BaseMetric",
    "EuclideanMetric",
    "MahalanobisMetric",
    "IoUMetric",
    "GIoUMetric",
    "MotionMetric",
    "HybridMetric",
]