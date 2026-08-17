"""
========================================================================
Validation Gating

Rejects impossible detection-to-track associations before assignment.

Supports

• Euclidean Gate
• Mahalanobis Gate

========================================================================
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult

from tracking.models.track import Track


class ValidationGate:

    """
    Validation gate used before assignment.
    """

    def __init__(

        self,

        distance_threshold: float = 75.0,

        mahalanobis_threshold: float = 9.21,

    ):

        self.distance_threshold = distance_threshold

        self.mahalanobis_threshold = mahalanobis_threshold

    ####################################################################
    # Euclidean Gate
    ####################################################################

    def euclidean(

        self,

        track: Track,

        detection: DetectionResult,

    ) -> bool:

        if track.latest_position is None:
            return False

        dx = detection.position.x - track.latest_position.x
        dy = detection.position.y - track.latest_position.y

        distance = np.hypot(dx, dy)

        return distance <= self.distance_threshold

    ####################################################################
    # Mahalanobis Gate
    ####################################################################

    def mahalanobis(

        self,

        track: Track,

        detection: DetectionResult,

    ) -> bool:

        if track.state is None:
            return False

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        z = np.array([
            detection.position.x,
            detection.position.y,
        ])

        x = track.state.state

        P = track.state.covariance

        R = np.eye(2) * 5.0

        innovation = z - H @ x

        S = H @ P @ H.T + R

        d2 = innovation.T @ np.linalg.inv(S) @ innovation

        return float(d2) <= self.mahalanobis_threshold

    ####################################################################
    # Combined Gate
    ####################################################################

    def validate(

        self,

        track: Track,

        detection: DetectionResult,

    ) -> bool:

        return (

            self.euclidean(track, detection)

            and

            self.mahalanobis(track, detection)

        )