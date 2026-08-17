"""
========================================================================
Base Association / Similarity Metric
========================================================================

Canonical metric interface used by the C-UAS tracking system.

Metrics convert a Track + DetectionResult pair into a scalar cost.

    Lower cost  -> Better association
    Higher cost -> Worse association

The metric layer is independent of the association algorithm.

Supported association algorithms:

    • Hungarian
    • Nearest Neighbor
    • JPDA
    • MHT

Typical metrics:

    • Euclidean
    • Mahalanobis
    • IoU
    • GIoU
    • Motion
    • Appearance
    • Hybrid

========================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import Any

from core.detection_result import DetectionResult
from tracking.models.track import Track


class BaseMetric(ABC):
    """
    Abstract base class for all tracking association metrics.

    Contract
    --------
    A metric receives:

        Track
        DetectionResult

    and returns:

        float

    Convention:

        smaller cost = better association
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        invalid_cost: float = 1e9,
    ) -> None:

        invalid_cost = float(invalid_cost)

        if not isfinite(invalid_cost):
            raise ValueError(
                "invalid_cost must be finite."
            )

        if invalid_cost <= 0.0:
            raise ValueError(
                "invalid_cost must be greater than zero."
            )

        self.invalid_cost = invalid_cost

    # ==================================================================
    # Compute
    # ==================================================================

    @abstractmethod
    def compute(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> float:
        """
        Compute the association cost.

        Parameters
        ----------
        track:
            Existing tracked object.

        detection:
            New sensor detection.

        Returns
        -------
        float
            Association cost.

        Notes
        -----
        Implementations should return ``self.invalid_cost`` when
        the pair cannot be meaningfully compared.

        Implementations must never return a negative cost.
        """

        raise NotImplementedError

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> None:
        """
        Validate common metric inputs.

        Subclasses can override this method for additional
        requirements, but should call ``super().validate()``.
        """

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        if not isinstance(
            detection,
            DetectionResult,
        ):
            raise TypeError(
                "detection must be a DetectionResult instance."
            )

    # ==================================================================
    # Safe Compute
    # ==================================================================

    def safe_compute(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> float:
        """
        Safely compute an association cost.

        Converts the following into ``invalid_cost``:

            • Invalid inputs
            • Exceptions
            • NaN
            • Positive infinity
            • Negative infinity

        Negative finite costs are also rejected.
        """

        try:

            self.validate(
                track,
                detection,
            )

            cost = self.compute(
                track,
                detection,
            )

            cost = float(cost)

            if not isfinite(cost):
                return self.invalid_cost

            if cost < 0.0:
                return self.invalid_cost

            return cost

        except Exception:
            return self.invalid_cost

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def name(self) -> str:
        """
        Human-readable metric name.

        Example
        -------
        EuclideanMetric -> "euclidean_metric"
        """

        return self.__class__.__name__

    # ==================================================================

    @property
    def requires_position(self) -> bool:
        """
        Whether the metric requires position information.

        Default:
            True
        """

        return True

    # ==================================================================

    @property
    def requires_velocity(self) -> bool:
        """
        Whether the metric requires velocity information.

        Default:
            False
        """

        return False

    # ==================================================================

    @property
    def requires_bbox(self) -> bool:
        """
        Whether the metric requires image bounding-box information.

        Default:
            False
        """

        return False

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict[str, Any]:
        """
        Return serializable metric configuration.
        """

        return {
            "name": self.name,
            "invalid_cost": self.invalid_cost,
            "requires_position": self.requires_position,
            "requires_velocity": self.requires_velocity,
            "requires_bbox": self.requires_bbox,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"invalid_cost={self.invalid_cost})"
        )