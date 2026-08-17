"""
========================================================================
3D Mahalanobis Association Metric
========================================================================

Computes the Mahalanobis distance between a predicted 3D Track state
and a 3D DetectionResult position.

State:

    [x, y, z, vx, vy, vz]

Measurement:

    [x, y, z]

Lower value = better association.

========================================================================
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


class MahalanobisMetric(BaseMetric):
    """
    3D Mahalanobis association metric.

    Uses the positional portion of a 6D constant-velocity state:

        [x, y, z, vx, vy, vz]

    Measurement:

        [x, y, z]

    The covariance of the state is propagated into measurement space
    before calculating the Mahalanobis distance.
    """

    def __init__(
        self,
        measurement_noise: float = 1.0,
        invalid_cost: float = 1e9,
    ) -> None:

        super().__init__(
            invalid_cost=invalid_cost
        )

        if measurement_noise <= 0:
            raise ValueError(
                "measurement_noise must be greater than zero."
            )

        self.measurement_noise = float(
            measurement_noise
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
        Compute 3D Mahalanobis distance.
        """

        if track.state_vector is None:
            return self.invalid_cost

        if detection.position is None:
            return self.invalid_cost

        state = track.state_vector

        # --------------------------------------------------------------
        # Validate state dimensions
        # --------------------------------------------------------------

        x = np.asarray(
            state.state,
            dtype=float,
        )

        P = np.asarray(
            state.covariance,
            dtype=float,
        )

        if x.shape != (6,):
            return self.invalid_cost

        if P.shape != (6, 6):
            return self.invalid_cost

        # --------------------------------------------------------------
        # Measurement model
        #
        # z = Hx
        #
        # State:
        #
        # [x, y, z, vx, vy, vz]
        #
        # Measurement:
        #
        # [x, y, z]
        # --------------------------------------------------------------

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Detection measurement
        # --------------------------------------------------------------

        z = np.array(
            [
                detection.position.x,
                detection.position.y,
                detection.position.z,
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Measurement noise
        # --------------------------------------------------------------

        R = (
            np.eye(
                3,
                dtype=float,
            )
            * self.measurement_noise
        )

        # --------------------------------------------------------------
        # Innovation
        # --------------------------------------------------------------

        innovation = z - H @ x

        # --------------------------------------------------------------
        # Innovation covariance
        # --------------------------------------------------------------

        S = (
            H
            @ P
            @ H.T
            + R
        )

        # --------------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------------

        try:

            solved = np.linalg.solve(
                S,
                innovation,
            )

        except np.linalg.LinAlgError:

            return self.invalid_cost

        distance_squared = float(
            innovation.T @ solved
        )

        if not np.isfinite(
            distance_squared
        ):
            return self.invalid_cost

        if distance_squared < 0:
            return self.invalid_cost

        return float(
            np.sqrt(distance_squared)
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        return True

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict:

        config = super().get_config()

        config.update(
            {
                "measurement_noise":
                    self.measurement_noise,

                "dimensions":
                    3,

                "state_dimension":
                    6,
            }
        )

        return config