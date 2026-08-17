"""
========================================================================
3D Constant-Acceleration Kalman Filter
========================================================================

Canonical Kalman filter for the C-UAS tracking framework.

State model:

    [x, y, z,
     vx, vy, vz,
     ax, ay, az]

Measurement model:

    [x, y, z]

The filter supports:

    • 3D position measurements
    • Prediction using StateCA
    • Kalman measurement update
    • Configurable measurement noise
    • Configurable jerk process noise
    • Innovation calculation
    • Mahalanobis distance
    • Numerical stability handling

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detection_result import DetectionResult
from tracking.models.state_ca import StateCA


# ======================================================================
# Kalman Filter
# ======================================================================

@dataclass(slots=True)
class KalmanFilter:

    """
    3D Constant-Acceleration Kalman Filter.

    State:

        [x, y, z,
         vx, vy, vz,
         ax, ay, az]

    Measurement:

        [x, y, z]
    """

    measurement_variance: float = 1.0

    jerk_variance: float = 1.0

    min_covariance: float = 1e-9

    # ==================================================================
    # Initialization
    # ==================================================================

    def __post_init__(self) -> None:

        if self.measurement_variance < 0.0:
            raise ValueError(
                "measurement_variance must be "
                "non-negative."
            )

        if self.jerk_variance < 0.0:
            raise ValueError(
                "jerk_variance must be "
                "non-negative."
            )

        if self.min_covariance <= 0.0:
            raise ValueError(
                "min_covariance must be "
                "greater than zero."
            )

    # ==================================================================
    # Constants
    # ==================================================================

    STATE_DIMENSION = 9

    MEASUREMENT_DIMENSION = 3

    # ==================================================================
    # Measurement Matrix
    # ==================================================================

    @staticmethod
    def measurement_matrix() -> np.ndarray:
        """
        Position-only measurement matrix.

        Maps:

            [x, y, z,
             vx, vy, vz,
             ax, ay, az]

        to:

            [x, y, z]
        """

        H = np.zeros(
            (3, 9),
            dtype=float,
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        return H

    # ==================================================================
    # Measurement Noise
    # ==================================================================

    def measurement_noise_matrix(
        self,
    ) -> np.ndarray:
        """
        Measurement noise covariance R.
        """

        return np.eye(
            self.MEASUREMENT_DIMENSION,
            dtype=float,
        ) * self.measurement_variance

    # ==================================================================
    # Prediction
    # ==================================================================

    def predict(
        self,
        state: StateCA,
        dt: float,
    ) -> StateCA:
        """
        Predict state forward by dt seconds.
        """

        if not isinstance(
            state,
            StateCA,
        ):
            raise TypeError(
                "state must be a StateCA instance."
            )

        if dt < 0.0:
            raise ValueError(
                "dt must be non-negative."
            )

        process_noise = (
            StateCA.constant_jerk_process_noise(
                dt=dt,
                jerk_variance=self.jerk_variance,
            )
        )

        predicted = state.predict(
            dt=dt,
            process_noise=process_noise,
        )

        predicted.covariance = (
            self._stabilize_covariance(
                predicted.covariance
            )
        )

        return predicted

    # ==================================================================
    # Update from Position
    # ==================================================================

    def update(
        self,
        state: StateCA,
        measurement: np.ndarray,
        measurement_noise: np.ndarray | None = None,
    ) -> StateCA:
        """
        Perform a Kalman measurement update.

        Parameters
        ----------
        state:
            Predicted StateCA.

        measurement:
            3-element position measurement:

                [x, y, z]

        measurement_noise:
            Optional 3x3 measurement covariance.
        """

        if not isinstance(
            state,
            StateCA,
        ):
            raise TypeError(
                "state must be a StateCA instance."
            )

        z = np.asarray(
            measurement,
            dtype=float,
        )

        if z.shape != (3,):
            raise ValueError(
                "measurement must have shape (3,)."
            )

        if not np.all(
            np.isfinite(z)
        ):
            raise ValueError(
                "measurement contains "
                "non-finite values."
            )

        H = self.measurement_matrix()

        R = (
            self.measurement_noise_matrix()
            if measurement_noise is None
            else np.asarray(
                measurement_noise,
                dtype=float,
            )
        )

        if R.shape != (3, 3):
            raise ValueError(
                "measurement_noise must have "
                "shape (3, 3)."
            )

        if not np.all(
            np.isfinite(R)
        ):
            raise ValueError(
                "measurement_noise contains "
                "non-finite values."
            )

        x = state.state.copy()

        P = state.covariance.copy()

        # --------------------------------------------------------------
        # Innovation
        # --------------------------------------------------------------

        innovation = (
            z - H @ x
        )

        # --------------------------------------------------------------
        # Innovation covariance
        # --------------------------------------------------------------

        S = (
            H
            @ P
            @ H.T
            + R
        )

        S = self._stabilize_covariance(
            S
        )

        # --------------------------------------------------------------
        # Kalman gain
        # --------------------------------------------------------------

        PHt = P @ H.T

        try:

            K = np.linalg.solve(
                S,
                PHt.T,
            ).T

        except np.linalg.LinAlgError:

            S = (
                S
                + np.eye(3)
                * self.min_covariance
            )

            K = np.linalg.solve(
                S,
                PHt.T,
            ).T

        # --------------------------------------------------------------
        # State update
        # --------------------------------------------------------------

        updated_state = (
            x
            + K @ innovation
        )

        # --------------------------------------------------------------
        # Joseph-form covariance update
        #
        # More numerically stable than:
        #
        #     P = (I-KH)P
        # --------------------------------------------------------------

        I = np.eye(
            self.STATE_DIMENSION,
            dtype=float,
        )

        IKH = (
            I
            - K @ H
        )

        updated_covariance = (
            IKH
            @ P
            @ IKH.T
            + K
            @ R
            @ K.T
        )

        updated_covariance = (
            self._stabilize_covariance(
                updated_covariance
            )
        )

        return StateCA(
            state=updated_state,
            covariance=updated_covariance,
        )

    # ==================================================================
    # Update from DetectionResult
    # ==================================================================

    def update_from_detection(
        self,
        state: StateCA,
        detection: DetectionResult,
        measurement_noise: np.ndarray | None = None,
    ) -> StateCA:
        """
        Update a StateCA using a DetectionResult.

        DetectionResult.position must contain
        a 3D Point3D.
        """

        if not isinstance(
            detection,
            DetectionResult,
        ):
            raise TypeError(
                "detection must be a "
                "DetectionResult instance."
            )

        if detection.position is None:
            raise ValueError(
                "DetectionResult does not "
                "contain a position."
            )

        measurement = np.array(
            [
                detection.position.x,
                detection.position.y,
                detection.position.z,
            ],
            dtype=float,
        )

        return self.update(
            state=state,
            measurement=measurement,
            measurement_noise=measurement_noise,
        )

    # ==================================================================
    # Innovation
    # ==================================================================

    def innovation(
        self,
        state: StateCA,
        measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate measurement innovation.
        """

        z = np.asarray(
            measurement,
            dtype=float,
        )

        if z.shape != (3,):
            raise ValueError(
                "measurement must have shape (3,)."
            )

        H = self.measurement_matrix()

        return (
            z
            - H @ state.state
        )

    # ==================================================================
    # Innovation Covariance
    # ==================================================================

    def innovation_covariance(
        self,
        state: StateCA,
        measurement_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate innovation covariance S.
        """

        H = self.measurement_matrix()

        R = (
            self.measurement_noise_matrix()
            if measurement_noise is None
            else np.asarray(
                measurement_noise,
                dtype=float,
            )
        )

        return (
            H
            @ state.covariance
            @ H.T
            + R
        )

    # ==================================================================
    # Mahalanobis Distance
    # ==================================================================

    def mahalanobis_distance(
        self,
        state: StateCA,
        measurement: np.ndarray,
        measurement_noise: np.ndarray | None = None,
    ) -> float:
        """
        Calculate squared-root Mahalanobis distance.
        """

        innovation = self.innovation(
            state,
            measurement,
        )

        S = self.innovation_covariance(
            state,
            measurement_noise,
        )

        S = self._stabilize_covariance(
            S
        )

        try:

            solved = np.linalg.solve(
                S,
                innovation,
            )

        except np.linalg.LinAlgError:

            S = (
                S
                + np.eye(3)
                * self.min_covariance
            )

            solved = np.linalg.solve(
                S,
                innovation,
            )

        distance_squared = (
            innovation.T
            @ solved
        )

        return float(
            np.sqrt(
                max(
                    0.0,
                    distance_squared,
                )
            )
        )

    # ==================================================================
    # Covariance Stabilization
    # ==================================================================

    def _stabilize_covariance(
        self,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Symmetrize and regularize covariance.
        """

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        covariance = (
            0.5
            * (
                covariance
                + covariance.T
            )
        )

        diagonal = np.diag(
            covariance
        )

        if np.any(
            diagonal < self.min_covariance
        ):

            covariance = covariance.copy()

            for index, value in enumerate(
                diagonal
            ):

                if value < self.min_covariance:

                    covariance[
                        index,
                        index
                    ] = self.min_covariance

        return covariance

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        state: StateCA,
    ) -> StateCA:
        """
        Return a defensive copy of the supplied state.
        """

        if not isinstance(
            state,
            StateCA,
        ):
            raise TypeError(
                "state must be a StateCA instance."
            )

        return StateCA(
            state=state.state.copy(),
            covariance=state.covariance.copy(),
        )

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "KalmanFilter("
            f"measurement_variance="
            f"{self.measurement_variance}, "
            f"jerk_variance="
            f"{self.jerk_variance})"
        )