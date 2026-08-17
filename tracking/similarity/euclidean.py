"""
========================================================================
3D Euclidean Association Metric
========================================================================

Computes the Euclidean distance between a Track position estimate
and a DetectionResult 3D position.

Canonical Track state:

    [x, y, z,
     vx, vy, vz,
     ax, ay, az]

Measurement:

    [x, y, z]

This metric intentionally uses position only.

Lower value
    -> better association

Higher value
    -> worse association

========================================================================
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from core.detection_result import DetectionResult

from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


class EuclideanMetric(BaseMetric):

    """
    3D Euclidean distance association metric.

    The metric compares the current Track position estimate
    against the 3D position reported by a detection.

    Priority of Track position:

        1. state_vector.position
        2. current_detection.position

    If neither is available, the association is invalid.
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        invalid_cost: float = 1e9,
    ) -> None:

        super().__init__(
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
        Compute 3D Euclidean distance.

        Parameters
        ----------
        track:
            Existing Track.

        detection:
            New DetectionResult.

        Returns
        -------
        float
            3D Euclidean distance.

        Returns ``invalid_cost`` if either side does not
        contain a valid 3D position.
        """

        # --------------------------------------------------------------
        # Detection validation
        # --------------------------------------------------------------

        if detection.position is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Track position
        # --------------------------------------------------------------

        track_position = track.latest_position

        if track_position is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Extract Track position
        # --------------------------------------------------------------

        try:
            if all(
                    hasattr(track_position, attr)
                    for attr in ("x", "y", "z")
            ):
                track_xyz = np.array(
                    [
                        track_position.x,
                        track_position.y,
                        track_position.z,
                    ],
                    dtype=float,
                )
            else:
                track_xyz = np.asarray(
                    track_position,
                    dtype=float,
                ).reshape(-1)

        except (
                TypeError,
                ValueError,
        ):
            return self.invalid_cost

        # --------------------------------------------------------------
        # Extract Detection position
        # --------------------------------------------------------------

        try:
            detection_position = detection.position

            if all(
                    hasattr(detection_position, attr)
                    for attr in ("x", "y", "z")
            ):
                detection_xyz = np.array(
                    [
                        detection_position.x,
                        detection_position.y,
                        detection_position.z,
                    ],
                    dtype=float,
                )
            else:
                detection_xyz = np.asarray(
                    detection_position,
                    dtype=float,
                ).reshape(-1)

        except (
                TypeError,
                ValueError,
        ):
            return self.invalid_cost

        # --------------------------------------------------------------
        # Validate dimensions
        # --------------------------------------------------------------

        if track_xyz.size != 3:
            return self.invalid_cost

        if detection_xyz.size != 3:
            return self.invalid_cost

        track_xyz = track_xyz.astype(
            float,
            copy=False,
        )

        detection_xyz = detection_xyz.astype(
            float,
            copy=False,
        )

        # --------------------------------------------------------------
        # Numerical validation
        # --------------------------------------------------------------

        if not np.all(
            np.isfinite(track_xyz)
        ):
            return self.invalid_cost

        if not np.all(
            np.isfinite(detection_xyz)
        ):
            return self.invalid_cost

        # --------------------------------------------------------------
        # Difference vector
        # --------------------------------------------------------------

        difference = (
            detection_xyz
            - track_xyz
        )

        # --------------------------------------------------------------
        # Euclidean distance
        #
        # d = sqrt(
        #       dx² +
        #       dy² +
        #       dz²
        #     )
        # --------------------------------------------------------------

        distance_squared = float(
            np.dot(
                difference,
                difference,
            )
        )

        if distance_squared < 0.0:
            return self.invalid_cost

        distance = sqrt(
            distance_squared
        )

        # --------------------------------------------------------------
        # Final numerical safety
        # --------------------------------------------------------------

        if not np.isfinite(
            distance
        ):
            return self.invalid_cost

        return float(
            distance
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        return True

    # ==================================================================

    def get_config(self) -> dict:

        config = super().get_config()

        config.update(
            {
                "dimensions": 3,

                "distance_type":
                    "euclidean",

                "state_model":
                    "constant_acceleration",

                "measurement_model":
                    "3d_position",
            }
        )

        return config

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "EuclideanMetric("
            f"invalid_cost="
            f"{self.invalid_cost})"
        )