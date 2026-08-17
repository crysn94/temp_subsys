"""
========================================================================
Canonical 3D State Vector
========================================================================

Canonical dynamic state representation used throughout the C-UAS
tracking framework.

Constant-Velocity State:

    x = [
        x,
        y,
        z,
        vx,
        vy,
        vz
    ]

Where:

    x, y, z       -> 3D position
    vx, vy, vz    -> 3D velocity

The state vector is accompanied by a 6x6 covariance matrix and a
UTC-aware Timestamp.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.point import Point3D
from core.geometry.velocity import Velocity3D
from core.timestamps import Timestamp


@dataclass(slots=True)
class StateVector:
    """
    Canonical 3D constant-velocity state vector.

    State ordering:

        [x, y, z, vx, vy, vz]

    Covariance:

        6 x 6

    Units depend on the coordinate system used by the tracking
    framework, but position and velocity must remain consistent.
    """

    ####################################################################
    # State
    ####################################################################

    state: np.ndarray

    ####################################################################
    # Uncertainty
    ####################################################################

    covariance: np.ndarray

    ####################################################################
    # Timestamp
    ####################################################################

    timestamp: Timestamp

    ####################################################################
    # Validation
    ####################################################################

    def __post_init__(self) -> None:
        """
        Validate and normalize the state vector and covariance.
        """

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        self.state = np.asarray(
            self.state,
            dtype=float,
        )

        if self.state.shape != (6,):
            raise ValueError(
                "StateVector state must have shape (6,) "
                "for [x, y, z, vx, vy, vz]."
            )

        if not np.all(
            np.isfinite(self.state)
        ):
            raise ValueError(
                "StateVector state contains non-finite values."
            )

        # --------------------------------------------------------------
        # Covariance
        # --------------------------------------------------------------

        self.covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if self.covariance.shape != (6, 6):
            raise ValueError(
                "StateVector covariance must have shape (6, 6)."
            )

        if not np.all(
            np.isfinite(self.covariance)
        ):
            raise ValueError(
                "StateVector covariance contains non-finite values."
            )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        if not isinstance(
            self.timestamp,
            Timestamp,
        ):
            raise TypeError(
                "timestamp must be a Timestamp instance."
            )

    ####################################################################
    # Position
    ####################################################################

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def z(self) -> float:
        return float(self.state[2])

    ####################################################################
    # Velocity
    ####################################################################

    @property
    def vx(self) -> float:
        return float(self.state[3])

    @property
    def vy(self) -> float:
        return float(self.state[4])

    @property
    def vz(self) -> float:
        return float(self.state[5])

    ####################################################################
    # Position Object
    ####################################################################

    @property
    def position(self) -> Point3D:
        """
        Return the position as the canonical Point3D object.
        """

        return Point3D(
            x=self.x,
            y=self.y,
            z=self.z,
        )

    ####################################################################
    # Velocity Object
    ####################################################################

    @property
    def velocity(self) -> Velocity3D:
        """
        Return the velocity as the canonical Velocity3D object.
        """

        return Velocity3D(
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
        )

    ####################################################################
    # Speed
    ####################################################################

    @property
    def speed(self) -> float:
        """
        Magnitude of the 3D velocity vector.
        """

        return float(
            np.linalg.norm(
                self.state[3:6]
            )
        )

    ####################################################################
    # State Components
    ####################################################################

    @property
    def position_vector(self) -> np.ndarray:
        """
        Return [x, y, z].
        """

        return self.state[:3].copy()

    @property
    def velocity_vector(self) -> np.ndarray:
        """
        Return [vx, vy, vz].
        """

        return self.state[3:6].copy()

    @property
    def dimension(self) -> int:
        """
        Number of elements in the state vector.

        Canonical C-UAS 3D constant-velocity state:

            [x, y, z, vx, vy, vz]

        Returns
        -------
        int
            State dimension.
        """
        return int(self.state.shape[0])

    ####################################################################
    # Array Conversion
    ####################################################################

    def to_array(self) -> np.ndarray:
        """
        Return a copy of the complete state vector.
        """

        return self.state.copy()

    ####################################################################
    # Factory
    ####################################################################

    @classmethod
    def from_components(
        cls,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        covariance: np.ndarray | None = None,
        timestamp: Timestamp | None = None,
    ) -> "StateVector":
        """
        Construct a StateVector from individual components.

        Parameters
        ----------
        x, y, z:
            Position.

        vx, vy, vz:
            Velocity.

        covariance:
            Optional 6x6 covariance matrix.

        timestamp:
            Optional Timestamp.

        Defaults
        -------
        covariance:
            Identity matrix.

        timestamp:
            Timestamp.now().
        """

        state = np.array(
            [
                x,
                y,
                z,
                vx,
                vy,
                vz,
            ],
            dtype=float,
        )

        if covariance is None:
            covariance = np.eye(
                6,
                dtype=float,
            )

        if timestamp is None:
            timestamp = Timestamp.now()

        return cls(
            state=state,
            covariance=np.asarray(
                covariance,
                dtype=float,
            ),
            timestamp=timestamp,
        )

    ####################################################################
    # Factory From Array
    ####################################################################

    @classmethod
    def from_array(
        cls,
        state: np.ndarray,
        covariance: np.ndarray,
        timestamp: Timestamp | None = None,
    ) -> "StateVector":
        """
        Construct a StateVector from arrays.
        """

        if timestamp is None:
            timestamp = Timestamp.now()

        return cls(
            state=np.asarray(
                state,
                dtype=float,
            ),
            covariance=np.asarray(
                covariance,
                dtype=float,
            ),
            timestamp=timestamp,
        )

    ####################################################################
    # Copy
    ####################################################################

    def copy(self) -> "StateVector":
        """
        Return a deep copy of the state vector.
        """

        return StateVector(
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            timestamp=self.timestamp,
        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self) -> dict:
        """
        Serialize the state vector.
        """

        return {
            "state": self.state.tolist(),

            "position": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            },

            "velocity": {
                "vx": self.vx,
                "vy": self.vy,
                "vz": self.vz,
            },

            "speed": self.speed,

            "covariance": self.covariance.tolist(),

            "timestamp": self.timestamp.as_dict(),
        }

    ####################################################################
    # Representation
    ####################################################################

    def __repr__(self) -> str:
        return (
            "StateVector("
            f"x={self.x:.3f}, "
            f"y={self.y:.3f}, "
            f"z={self.z:.3f}, "
            f"vx={self.vx:.3f}, "
            f"vy={self.vy:.3f}, "
            f"vz={self.vz:.3f})"
        )