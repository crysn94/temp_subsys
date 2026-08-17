"""
========================================================================
Hybrid Association Metric
========================================================================

Combines multiple association metrics into one configurable cost.

The HybridMetric provides a common association score for the
multi-sensor C-UAS tracking system.

Supported components
--------------------

    • EuclideanMetric
    • MahalanobisMetric
    • MotionMetric
    • IoUMetric
    • GIoUMetric

The metric follows the canonical convention:

    lower cost = better association
    higher cost = worse association

Typical architecture
--------------------

                    HybridMetric
                         |
          +--------------+--------------+
          |              |              |
      Position         Motion        Geometry
          |              |              |
    Mahalanobis       MotionMetric    IoU/GIoU
          |
     3D State +
     covariance

The metric is independent of the assignment algorithm and can
therefore be used with:

    • Hungarian
    • Nearest Neighbor
    • JPDA
    • MHT

========================================================================
"""

from __future__ import annotations

from typing import Any

from core.detection_result import DetectionResult

from tracking.models.track import Track

from tracking.similarity.base_metric import BaseMetric
from tracking.similarity.euclidean import EuclideanMetric
from tracking.similarity.giou import GIoUMetric
from tracking.similarity.iou import IoUMetric
from tracking.similarity import MahalanobisMetric
from tracking.similarity.motion import MotionMetric


class HybridMetric(BaseMetric):
    """
    Weighted combination of multiple association metrics.

    The default configuration combines:

        Mahalanobis
        Motion
        IoU
        GIoU

    Each component contributes:

        weight × metric_cost

    The final cost is the weighted average of the valid
    component costs.

    Parameters
    ----------
    position_weight:
        Weight assigned to Mahalanobis position cost.

    motion_weight:
        Weight assigned to MotionMetric.

    iou_weight:
        Weight assigned to IoU cost.

    giou_weight:
        Weight assigned to GIoU cost.

    invalid_cost:
        Cost returned when no usable metric is available.
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        position_weight: float = 0.40,
        motion_weight: float = 0.25,
        iou_weight: float = 0.20,
        giou_weight: float = 0.15,
        invalid_cost: float = 1e9,
    ) -> None:

        super().__init__(
            invalid_cost=invalid_cost
        )

        weights = {
            "position": position_weight,
            "motion": motion_weight,
            "iou": iou_weight,
            "giou": giou_weight,
        }

        for name, weight in weights.items():

            if weight < 0.0:
                raise ValueError(
                    f"{name}_weight must be non-negative."
                )

        total_weight = sum(
            weights.values()
        )

        if total_weight <= 0.0:
            raise ValueError(
                "At least one HybridMetric weight "
                "must be greater than zero."
            )

        # --------------------------------------------------------------
        # Store normalized weights
        # --------------------------------------------------------------

        self.position_weight = (
            position_weight / total_weight
        )

        self.motion_weight = (
            motion_weight / total_weight
        )

        self.iou_weight = (
            iou_weight / total_weight
        )

        self.giou_weight = (
            giou_weight / total_weight
        )

        # --------------------------------------------------------------
        # Component metrics
        # --------------------------------------------------------------

        self.position_metric = MahalanobisMetric(
            invalid_cost=invalid_cost
        )

        self.motion_metric = MotionMetric(
            invalid_cost=invalid_cost
        )

        self.iou_metric = IoUMetric(
            invalid_cost=invalid_cost
        )

        self.giou_metric = GIoUMetric(
            invalid_cost=invalid_cost
        )

    # ==================================================================
    # Compute
    # ==================================================================

    def compute(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> float:
        """
        Compute the combined association cost.

        Only metrics whose required information is available
        contribute to the final score.

        This is important for C-UAS multi-sensor operation because
        not every sensor produces the same measurement types.

        For example:

            Radar:
                position + velocity

            EO/IR:
                position + bounding box

            RF:
                position may be available while velocity
                or bounding-box information may not be available.
        """

        costs: list[tuple[float, float]] = []

        # --------------------------------------------------------------
        # Mahalanobis / position
        # --------------------------------------------------------------

        if (
            self.position_weight > 0.0
            and track.state_vector is not None
            and detection.position is not None
        ):

            cost = self.position_metric.safe_compute(
                track,
                detection,
            )

            if cost < self.invalid_cost:

                costs.append(
                    (
                        self.position_weight,
                        cost,
                    )
                )

        # --------------------------------------------------------------
        # Motion
        # --------------------------------------------------------------

        if (
            self.motion_weight > 0.0
            and track.state_vector is not None
            and detection.velocity is not None
        ):

            cost = self.motion_metric.safe_compute(
                track,
                detection,
            )

            if cost < self.invalid_cost:

                costs.append(
                    (
                        self.motion_weight,
                        cost,
                    )
                )

        # --------------------------------------------------------------
        # IoU
        # --------------------------------------------------------------

        if (
            self.iou_weight > 0.0
            and track.current_detection is not None
            and detection.bbox is not None
        ):

            cost = self.iou_metric.safe_compute(
                track,
                detection,
            )

            if cost < self.invalid_cost:

                costs.append(
                    (
                        self.iou_weight,
                        cost,
                    )
                )

        # --------------------------------------------------------------
        # GIoU
        # --------------------------------------------------------------

        if (
            self.giou_weight > 0.0
            and track.current_detection is not None
            and detection.bbox is not None
        ):

            cost = self.giou_metric.safe_compute(
                track,
                detection,
            )

            if cost < self.invalid_cost:

                costs.append(
                    (
                        self.giou_weight,
                        cost,
                    )
                )

        # --------------------------------------------------------------
        # No valid metric
        # --------------------------------------------------------------

        if not costs:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Weighted average
        # --------------------------------------------------------------

        total_weight = sum(
            weight
            for weight, _ in costs
        )

        if total_weight <= 0.0:
            return self.invalid_cost

        total_cost = sum(
            weight * cost
            for weight, cost in costs
        )

        return float(
            total_cost / total_weight
        )

    # ==================================================================
    # Requirements
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        """
        HybridMetric can operate without position if another
        configured metric has sufficient information.
        """

        return (
            self.position_weight > 0.0
        )

    # ==================================================================

    @property
    def requires_velocity(self) -> bool:
        """
        Velocity is required only when the motion component
        has a non-zero weight.
        """

        return (
            self.motion_weight > 0.0
        )

    # ==================================================================

    @property
    def requires_bbox(self) -> bool:
        """
        Bounding-box information is required when either
        IoU or GIoU contributes to the metric.
        """

        return (
            self.iou_weight > 0.0
            or self.giou_weight > 0.0
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict[str, Any]:
        """
        Return complete serializable configuration.
        """

        config = super().get_config()

        config.update(
            {
                "weights": {
                    "position":
                        self.position_weight,

                    "motion":
                        self.motion_weight,

                    "iou":
                        self.iou_weight,

                    "giou":
                        self.giou_weight,
                },

                "components": {
                    "position":
                        self.position_metric.name,

                    "motion":
                        self.motion_metric.name,

                    "iou":
                        self.iou_metric.name,

                    "giou":
                        self.giou_metric.name,
                },
            }
        )

        return config

    # ==================================================================
    # Component Metrics
    # ==================================================================

    @property
    def metrics(self) -> dict[str, BaseMetric]:
        """
        Return component metrics.

        Useful for diagnostics and testing.
        """

        return {
            "position":
                self.position_metric,

            "motion":
                self.motion_metric,

            "iou":
                self.iou_metric,

            "giou":
                self.giou_metric,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "HybridMetric("
            f"position={self.position_weight:.3f}, "
            f"motion={self.motion_weight:.3f}, "
            f"iou={self.iou_weight:.3f}, "
            f"giou={self.giou_weight:.3f})"
        )