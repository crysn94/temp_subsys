"""
========================================================================
Tracking Association
========================================================================

Association algorithms used by the C-UAS tracking framework.

========================================================================
"""

from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.association.hungarian import (
    AssociationResult,
    HungarianAssociator,
)

__all__ = [
    "AssociationResult",
    "CostMatrixBuilder",
    "HungarianAssociator",
]