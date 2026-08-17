"""
========================================================================
3D Motion Association Metric
========================================================================

Computes motion similarity between a Track and DetectionResult.

The canonical tracking state is:

    [x, y, z, vx, vy, vz]

This metric compares the velocity components:

    [vx, vy, vz]

Lower value
    Better association

Higher value
    Worse association

Used by
--------
• Data Association
• Hungarian Assignment
• Nearest Neighbor
• JPDA
• MHT
• Hybrid Association Metrics

========================================================================
"""

from __future__ import annotations

from math import sqrt

from core.detection_result import DetectionResult
from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


class MotionMetric(BaseMetric):
    """
    3D velocity-based association metric.

    The metric computes the Euclidean distance between the
    track's estimated velocity and the detection's measured
    velocity.

    Velocity vectors:

        Track:
            [vx, vy, vz]

        Detection:
            [vx, vy, vz]

    Cost:

        sqrt(
            (vx_t - vx_d)^2 +
            (vy_t - vy_d)^2 +
            (vz_t - vz_d)^2
        )

    Therefore:

        smaller cost = better motion match
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        velocity_scale: float = 1.0,
        invalid_cost: float = 1e9,
    ) -> None:

        super().__init__(
            invalid_cost=invalid_cost
        )

        if velocity_scale <= 0.0:
            raise ValueError(
                "velocity_scale must be greater than zero."
            )

        self.velocity_scale = float(
            velocity_scale
        )

    # ==================================================================
    # Compute
    # ==================================================================

    def compute(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> float:

        # --------------------------------------------------------------
        # Track state validation
        # --------------------------------------------------------------

        if track.state_vector is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Detection velocity validation
        # --------------------------------------------------------------

        if detection.velocity is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Track velocity
        # --------------------------------------------------------------

        track_velocity = track.state_vector.velocity

        # --------------------------------------------------------------
        # Detection velocity
        # --------------------------------------------------------------

        detection_velocity = detection.velocity

        # --------------------------------------------------------------
        # Velocity difference
        # --------------------------------------------------------------

        dvx = (
            track_velocity.vx
            - detection_velocity.vx
        )

        dvy = (
            track_velocity.vy
            - detection_velocity.vy
        )

        dvz = (
            track_velocity.vz
            - detection_velocity.vz
        )

        # --------------------------------------------------------------
        # 3D Euclidean velocity distance
        # --------------------------------------------------------------

        distance = sqrt(
            dvx * dvx
            +
            dvy * dvy
            +
            dvz * dvz
        )

        # --------------------------------------------------------------
        # Normalize by configured velocity scale
        # --------------------------------------------------------------

        cost = distance / self.velocity_scale

        if cost < 0.0:
            return self.invalid_cost

        return float(cost)

    # ==================================================================
    # Metric Requirements
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        """
        MotionMetric does not require position.

        It operates purely on velocity.
        """

        return False

    # ==================================================================

    @property
    def requires_velocity(self) -> bool:
        """
        MotionMetric requires velocity information.
        """

        return True

    # ==================================================================

    # Configuration
    # ==================================================================

    def get_config(self) -> dict:

        config = super().get_config()

        config.update(
            {
                "velocity_scale":
                    self.velocity_scale,

                "dimensions":
                    3,

                "comparison":
                    "velocity_euclidean_distance",
            }
        )

        return config

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            f"MotionMetric("
            f"velocity_scale={self.velocity_scale}, "
            f"invalid_cost={self.invalid_cost})"
        )