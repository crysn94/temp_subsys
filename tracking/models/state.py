"""
========================================================================
Dynamic State Interface
========================================================================

Canonical abstract interface for dynamic tracking states.

A DynamicState represents the estimated kinematic state of a tracked
object.

The interface is intentionally independent of the specific motion model.

Supported implementations include:

    • StateVector  -> 3D Constant Velocity
    • StateCA      -> 3D Constant Acceleration
    • Future models such as Coordinated Turn / IMM states

Canonical 3D representation:

    Position:
        [x, y, z]

    Velocity:
        [vx, vy, vz]

========================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.geometry import Point3D, Velocity3D


class DynamicState(ABC):
    """
    Abstract base class for all dynamic tracking states.

    The Track object depends on this interface rather than on a
    particular motion model.
    """

    # ==================================================================
    # Position
    # ==================================================================

    @property
    @abstractmethod
    def position(self) -> Point3D:
        """
        Current estimated 3D position.
        """
        raise NotImplementedError

    # ==================================================================
    # Velocity
    # ==================================================================

    @property
    @abstractmethod
    def velocity(self) -> Velocity3D:
        """
        Current estimated 3D velocity.
        """
        raise NotImplementedError

    # ==================================================================
    # Speed
    # ==================================================================

    @property
    def speed(self) -> float:
        """
        Current estimated speed.
        """

        return self.velocity.speed

    # ==================================================================
    # Model Identification
    # ==================================================================

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Human-readable identifier for the dynamic model.
        """
        raise NotImplementedError

    # ==================================================================
    # State Vector
    # ==================================================================

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """
        Raw numerical state vector.
        """
        raise NotImplementedError

    # ==================================================================
    # Covariance
    # ==================================================================

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        State covariance matrix.
        """
        raise NotImplementedError

    # ==================================================================
    # Dimensions
    # ==================================================================

    @property
    def state_dimension(self) -> int:
        """
        Number of elements in the state vector.
        """

        return int(self.state.shape[0])

    # ==================================================================
    # Position Availability
    # ==================================================================

    @property
    def has_position(self) -> bool:
        return self.position is not None

    # ==================================================================
    # Velocity Availability
    # ==================================================================

    @property
    def has_velocity(self) -> bool:
        return self.velocity is not None

    # ==================================================================
    # Prediction
    # ==================================================================

    @abstractmethod
    def predict(
        self,
        dt: float,
        process_noise: np.ndarray | None = None,
    ) -> "DynamicState":
        """
        Predict the state forward by dt seconds.

        Parameters
        ----------
        dt:
            Time interval in seconds.

        process_noise:
            Optional process-noise covariance.

        Returns
        -------
        DynamicState
            Predicted state.
        """

        raise NotImplementedError

    # ==================================================================
    # Copy
    # ==================================================================

    @abstractmethod
    def copy(self) -> "DynamicState":
        """
        Return an independent copy of the state.
        """
        raise NotImplementedError

    # ==================================================================
    # Serialization
    # ==================================================================

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable representation.
        """
        raise NotImplementedError

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(self) -> None:
        """
        Validate the dynamic state.

        Subclasses may extend this method.
        """

        state = np.asarray(
            self.state,
            dtype=float,
        )

        covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if state.ndim != 1:
            raise ValueError(
                "Dynamic state must be a one-dimensional vector."
            )

        if covariance.ndim != 2:
            raise ValueError(
                "Dynamic state covariance must be a matrix."
            )

        if covariance.shape != (
            state.shape[0],
            state.shape[0],
        ):
            raise ValueError(
                "Covariance dimensions must match "
                "the state vector dimensions."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                "Dynamic state contains non-finite values."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "Dynamic state covariance contains "
                "non-finite values."
            )

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"dimension={self.state_dimension}, "
            f"position={self.position}, "
            f"velocity={self.velocity})"
        )