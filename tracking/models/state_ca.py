"""
========================================================================
3D Constant Acceleration State
========================================================================

Canonical 3D Constant Acceleration (CA) dynamic state representation.

State vector:

    [x, y, z,
     vx, vy, vz,
     ax, ay, az]

where:

    x, y, z       -> position
    vx, vy, vz    -> velocity
    ax, ay, az    -> acceleration

Covariance:

    9 x 9

Used by:

    • Kalman Filter
    • Sensor Fusion
    • Trajectory Prediction
    • Threat Assessment
    • Intercept Prediction

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry import Point3D
from core.geometry import Velocity3D
from core.geometry.acceleration import Acceleration3D

from tracking.models.state import DynamicState


@dataclass(slots=True)
class StateCA(DynamicState):
    """
    3D Constant Acceleration state.

    State ordering:

        [x, y, z,
         vx, vy, vz,
         ax, ay, az]

    Position:
        indices 0:3

    Velocity:
        indices 3:6

    Acceleration:
        indices 6:9

    Covariance:
        9 x 9
    """

    state: np.ndarray
    covariance: np.ndarray

    # ==================================================================
    # Initialization
    # ==================================================================

    def __post_init__(self) -> None:

        self.state = np.asarray(
            self.state,
            dtype=float,
        )

        self.covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if self.state.shape != (9,):
            raise ValueError(
                "StateCA state must have shape (9,). "
                "Expected "
                "[x, y, z, vx, vy, vz, ax, ay, az]."
            )

        if self.covariance.shape != (9, 9):
            raise ValueError(
                "StateCA covariance must have shape (9, 9)."
            )

        if not np.all(np.isfinite(self.state)):
            raise ValueError(
                "StateCA state contains non-finite values."
            )

        if not np.all(
            np.isfinite(self.covariance)
        ):
            raise ValueError(
                "StateCA covariance contains "
                "non-finite values."
            )

    # ==================================================================
    # Model
    # ==================================================================

    @property
    def model_name(self) -> str:
        return "constant_acceleration_3d"

    # ==================================================================
    # Position
    # ==================================================================

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def z(self) -> float:
        return float(self.state[2])

    @property
    def position(self) -> Point3D:

        return Point3D(
            x=self.x,
            y=self.y,
            z=self.z,
        )

    # ==================================================================
    # Velocity
    # ==================================================================

    @property
    def vx(self) -> float:
        return float(self.state[3])

    @property
    def vy(self) -> float:
        return float(self.state[4])

    @property
    def vz(self) -> float:
        return float(self.state[5])

    @property
    def velocity(self) -> Velocity3D:

        return Velocity3D(
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
        )

    @property
    def speed(self) -> float:
        return self.velocity.speed

    # ==================================================================
    # Acceleration
    # ==================================================================

    @property
    def ax(self) -> float:
        return float(self.state[6])

    @property
    def ay(self) -> float:
        return float(self.state[7])

    @property
    def az(self) -> float:
        return float(self.state[8])

    @property
    def acceleration(self) -> Acceleration3D:

        return Acceleration3D(
            ax=self.ax,
            ay=self.ay,
            az=self.az,
        )

    # ==================================================================
    # State Components
    # ==================================================================

    @property
    def position_vector(self) -> np.ndarray:

        return self.state[0:3].copy()

    @property
    def velocity_vector(self) -> np.ndarray:

        return self.state[3:6].copy()

    @property
    def acceleration_vector(self) -> np.ndarray:

        return self.state[6:9].copy()

    # ==================================================================
    # Covariance Components
    # ==================================================================

    @property
    def position_covariance(self) -> np.ndarray:

        return self.covariance[
            0:3,
            0:3,
        ].copy()

    @property
    def velocity_covariance(self) -> np.ndarray:

        return self.covariance[
            3:6,
            3:6,
        ].copy()

    @property
    def acceleration_covariance(self) -> np.ndarray:

        return self.covariance[
            6:9,
            6:9,
        ].copy()

    # ==================================================================
    # Prediction
    # ==================================================================

    def predict(
        self,
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> "StateCA":
        """
        Predict the state forward using the 3D constant acceleration
        model.

        Position:

            x' = x + vx*dt + 0.5*ax*dt²

        Velocity:

            vx' = vx + ax*dt

        Acceleration:

            ax' = ax

        The same model is applied independently to X, Y and Z.
        """

        if dt < 0:
            raise ValueError(
                "dt must be non-negative."
            )

        dt2 = dt * dt

        # --------------------------------------------------------------
        # State transition matrix
        # --------------------------------------------------------------

        F = np.array(
            [
                # x
                [1.0, 0.0, 0.0,
                 dt,  0.0, 0.0,
                 0.5 * dt2, 0.0, 0.0],

                # y
                [0.0, 1.0, 0.0,
                 0.0, dt,  0.0,
                 0.0, 0.5 * dt2, 0.0],

                # z
                [0.0, 0.0, 1.0,
                 0.0, 0.0, dt,
                 0.0, 0.0, 0.5 * dt2],

                # vx
                [0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0,
                 dt,  0.0, 0.0],

                # vy
                [0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, dt,  0.0],

                # vz
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0,
                 0.0, 0.0, dt],

                # ax
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0],

                # ay
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0],

                # az
                [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Predicted state
        # --------------------------------------------------------------

        predicted_state = (
            F @ self.state
        )

        # --------------------------------------------------------------
        # Predicted covariance
        # --------------------------------------------------------------

        predicted_covariance = (
            F
            @ self.covariance
            @ F.T
        )

        # --------------------------------------------------------------
        # Optional process noise
        # --------------------------------------------------------------

        if process_noise is not None:

            process_noise = np.asarray(
                process_noise,
                dtype=float,
            )

            if process_noise.shape != (9, 9):
                raise ValueError(
                    "process_noise must have shape (9, 9)."
                )

            if not np.all(
                np.isfinite(process_noise)
            ):
                raise ValueError(
                    "process_noise contains "
                    "non-finite values."
                )

            predicted_covariance += (
                process_noise
            )

        return StateCA(
            state=predicted_state,
            covariance=predicted_covariance,
        )

    # ==================================================================
    # Process Noise
    # ==================================================================

    @staticmethod
    def constant_jerk_process_noise(
        dt: float,
        jerk_variance: float,
    ) -> np.ndarray:
        """
        Construct a 3D process-noise covariance matrix based on
        white jerk noise.

        This is useful when the CA model is driven by an assumed
        random jerk.

        Parameters
        ----------
        dt:
            Time step in seconds.

        jerk_variance:
            Variance of jerk noise.

        Returns
        -------
        np.ndarray
            9 x 9 process-noise covariance.
        """

        if dt < 0:
            raise ValueError(
                "dt must be non-negative."
            )

        if jerk_variance < 0:
            raise ValueError(
                "jerk_variance must be "
                "non-negative."
            )

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5

        q = jerk_variance

        # Single-axis CA process noise:
        #
        # [position, velocity, acceleration]
        #
        # driven by white jerk.

        Q_axis = q * np.array(
            [
                [
                    dt5 / 20.0,
                    dt4 / 8.0,
                    dt3 / 6.0,
                ],
                [
                    dt4 / 8.0,
                    dt3 / 3.0,
                    dt2 / 2.0,
                ],
                [
                    dt3 / 6.0,
                    dt2 / 2.0,
                    dt,
                ],
            ],
            dtype=float,
        )

        Q = np.zeros(
            (9, 9),
            dtype=float,
        )

        # X
        indices_x = [0, 3, 6]

        # Y
        indices_y = [1, 4, 7]

        # Z
        indices_z = [2, 5, 8]

        for indices in (
            indices_x,
            indices_y,
            indices_z,
        ):

            for i, row in enumerate(indices):

                for j, col in enumerate(indices):

                    Q[row, col] = Q_axis[i, j]

        return Q

    # ==================================================================
    # Copy
    # ==================================================================

    def copy(self) -> "StateCA":

        return StateCA(
            state=self.state.copy(),
            covariance=self.covariance.copy(),
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def as_dict(self) -> dict:

        return {
            "model": self.model_name,

            "state": self.state.tolist(),

            "position":
                self.position.as_dict(),

            "velocity":
                self.velocity.as_tuple(),

            "acceleration":
                self.acceleration.as_tuple(),

            "speed":
                self.speed,

            "covariance":
                self.covariance.tolist(),
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "StateCA("
            f"position={self.position}, "
            f"velocity={self.velocity}, "
            f"acceleration={self.acceleration}, "
            f"model='{self.model_name}'"
            ")"
        )