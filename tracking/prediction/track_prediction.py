"""
========================================================================
Track Prediction
========================================================================

3D Constant-Velocity Track Prediction.

State:

    [x, y, z, vx, vy, vz]

Prediction:

    x'  = x + vx * dt
    y'  = y + vy * dt
    z'  = z + vz * dt

    vx' = vx
    vy' = vy
    vz' = vz

The predictor is NON-MUTATING.

It returns a new StateVector and never modifies the Track or its
original StateVector.

Responsibilities
----------------
- Predict a Track forward in time
- Preserve velocity
- Propagate covariance
- Accept optional process noise
- Reject invalid dt
- Preserve Track identity
- Provide deterministic predictions
========================================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tracking.models.state_vector import StateVector
from tracking.models.track import Track


# ======================================================================
# Constants
# ======================================================================

STATE_DIMENSION = 6


# ======================================================================
# Track Prediction
# ======================================================================


class TrackPrediction:
    """
    Predict Track state using a 3D constant-velocity model.

    Prediction is non-mutating.
    """

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(self) -> None:
        """
        Construct a TrackPrediction instance.
        """
        pass

    # ==================================================================
    # Main prediction
    # ==================================================================

    def predict(
        self,
        track: Track,
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> StateVector:
        """
        Predict a Track forward by ``dt`` seconds.

        Parameters
        ----------
        track:
            Track to predict.

        dt:
            Prediction interval in seconds.

        process_noise:
            Optional 6x6 process-noise covariance.

        Returns
        -------
        StateVector
            New predicted StateVector.

        Notes
        -----
        The original Track and StateVector are never modified.
        """

        # --------------------------------------------------------------
        # Validate Track
        # --------------------------------------------------------------

        self._validate_track(track)

        # --------------------------------------------------------------
        # Validate dt
        # --------------------------------------------------------------

        dt = self._validate_dt(dt)

        # --------------------------------------------------------------
        # Retrieve current state
        # --------------------------------------------------------------

        state = track.state_vector

        if state is None:
            raise ValueError(
                "Track must contain a state_vector."
            )

        if not isinstance(state, StateVector):
            raise TypeError(
                "TrackPrediction currently supports "
                "StateVector constant-velocity states."
            )

        # --------------------------------------------------------------
        # Validate process noise
        # --------------------------------------------------------------

        Q = self._validate_process_noise(
            process_noise
        )

        # --------------------------------------------------------------
        # State transition matrix
        #
        # State:
        #
        # [x, y, z, vx, vy, vz]
        # --------------------------------------------------------------

        F = np.array(
            [
                [1.0, 0.0, 0.0, dt,  0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt,  0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],

                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Copy original state
        #
        # This guarantees that prediction cannot modify the Track.
        # --------------------------------------------------------------

        original_state = np.asarray(
            state.state,
            dtype=float,
        ).copy()

        original_covariance = np.asarray(
            state.covariance,
            dtype=float,
        ).copy()

        # --------------------------------------------------------------
        # Predict state
        # --------------------------------------------------------------

        predicted_state = (
            F @ original_state
        )

        # --------------------------------------------------------------
        # Predict covariance
        # --------------------------------------------------------------

        predicted_covariance = (
            F
            @ original_covariance
            @ F.T
        )

        # --------------------------------------------------------------
        # Add process noise
        # --------------------------------------------------------------

        if Q is not None:
            predicted_covariance = (
                predicted_covariance + Q
            )

        # --------------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------------

        predicted_state = np.asarray(
            predicted_state,
            dtype=float,
        )

        predicted_covariance = np.asarray(
            predicted_covariance,
            dtype=float,
        )

        if not np.all(
            np.isfinite(predicted_state)
        ):
            raise ValueError(
                "Predicted state contains non-finite values."
            )

        if not np.all(
            np.isfinite(predicted_covariance)
        ):
            raise ValueError(
                "Predicted covariance contains non-finite values."
            )

        # --------------------------------------------------------------
        # Force covariance symmetry.
        # --------------------------------------------------------------

        predicted_covariance = (
            predicted_covariance
            + predicted_covariance.T
        ) / 2.0

        # --------------------------------------------------------------
        # Create a NEW StateVector.
        #
        # Important:
        # Do not call track.update_state().
        # --------------------------------------------------------------

        return StateVector(
            state=predicted_state,
            covariance=predicted_covariance,
            timestamp=state.timestamp,
        )

    # ==================================================================
    # Alias
    # ==================================================================

    def predict_track(
        self,
        track: Track,
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> StateVector:
        """
        Alias for predict().
        """

        return self.predict(
            track=track,
            dt=dt,
            process_noise=process_noise,
        )

    # ==================================================================
    # Multiple tracks
    # ==================================================================

    def predict_tracks(
        self,
        tracks: list[Track],
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> list[StateVector]:
        """
        Predict multiple tracks.

        Each returned StateVector corresponds to one Track.
        """

        if not isinstance(
            tracks,
            list,
        ):
            raise TypeError(
                "tracks must be a list of Track objects."
            )

        return [
            self.predict(
                track=track,
                dt=dt,
                process_noise=process_noise,
            )
            for track in tracks
        ]

    # ==================================================================
    # State prediction alias
    # ==================================================================

    def predict_state(
        self,
        track: Track,
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> StateVector:
        """
        Alias for predict().
        """

        return self.predict(
            track=track,
            dt=dt,
            process_noise=process_noise,
        )

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_track(
        track: Track,
    ) -> None:
        """
        Validate Track.
        """

        if not isinstance(
            track,
            Track,
        ):
            raise TypeError(
                "track must be a Track instance."
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dt(
        dt: float,
    ) -> float:
        """
        Validate prediction interval.
        """

        if not isinstance(
            dt,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            raise TypeError(
                "dt must be a numeric value."
            )

        dt = float(dt)

        if not np.isfinite(dt):
            raise ValueError(
                "dt must be finite."
            )

        if dt < 0.0:
            raise ValueError(
                "dt must be non-negative."
            )

        return dt

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_process_noise(
        process_noise: np.ndarray | None,
    ) -> np.ndarray | None:
        """
        Validate optional 6x6 process-noise covariance.
        """

        if process_noise is None:
            return None

        Q = np.asarray(
            process_noise,
            dtype=float,
        )

        if Q.shape != (
            STATE_DIMENSION,
            STATE_DIMENSION,
        ):
            raise ValueError(
                "process_noise must be a 6x6 matrix."
            )

        if not np.all(
            np.isfinite(Q)
        ):
            raise ValueError(
                "process_noise must contain only "
                "finite values."
            )

        if not np.allclose(
            Q,
            Q.T,
        ):
            raise ValueError(
                "process_noise must be symmetric."
            )

        return Q.copy()

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict[str, Any]:
        """
        Return predictor configuration.
        """

        return {
            "algorithm": "constant_velocity",
            "dimension": STATE_DIMENSION,
            "state_order": [
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
            ],
            "mutates_track": False,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:
        return (
            "TrackPrediction("
            "model='constant_velocity_3d', "
            "mutates_track=False"
            ")"
        )