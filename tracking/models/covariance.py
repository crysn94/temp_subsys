"""
========================================================================
Tracking Covariance Model
========================================================================

Canonical covariance utilities for the C-UAS tracking framework.

Primary state model:

    [x, y, z,
     vx, vy, vz,
     ax, ay, az]

Default covariance size:

    9 x 9

Used by:

    • Kalman Filter
    • Extended Kalman Filter
    • Unscented Kalman Filter
    • IMM Filter
    • Sensor Fusion
    • Data Association
    • Trajectory Prediction
    • Threat Assessment

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ======================================================================
# Covariance
# ======================================================================

@dataclass(slots=True)
class Covariance:
    """
    Covariance matrix associated with a tracking state.

    Default state dimension is 9:

        [x, y, z,
         vx, vy, vz,
         ax, ay, az]

    The class provides validation and convenient access to
    position, velocity and acceleration uncertainty.
    """

    matrix: np.ndarray

    # ==================================================================
    # Initialization
    # ==================================================================

    def __post_init__(self) -> None:

        self.matrix = np.asarray(
            self.matrix,
            dtype=float,
        )

        if self.matrix.ndim != 2:
            raise ValueError(
                "Covariance matrix must be 2-dimensional."
            )

        if (
            self.matrix.shape[0]
            != self.matrix.shape[1]
        ):
            raise ValueError(
                "Covariance matrix must be square."
            )

        if not np.all(
            np.isfinite(self.matrix)
        ):
            raise ValueError(
                "Covariance matrix contains "
                "non-finite values."
            )

    # ==================================================================
    # Dimension
    # ==================================================================

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    # ==================================================================
    # Shape
    # ==================================================================

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    # ==================================================================
    # Position Covariance
    # ==================================================================

    @property
    def position(self) -> np.ndarray:
        """
        3x3 position covariance.

        State indices:

            x -> 0
            y -> 1
            z -> 2
        """

        if self.dimension < 3:
            raise ValueError(
                "Covariance dimension must be at least 3."
            )

        return self.matrix[
            0:3,
            0:3,
        ].copy()

    # ==================================================================
    # Velocity Covariance
    # ==================================================================

    @property
    def velocity(self) -> np.ndarray:
        """
        3x3 velocity covariance.

        State indices:

            vx -> 3
            vy -> 4
            vz -> 5
        """

        if self.dimension < 6:
            raise ValueError(
                "Covariance dimension must be at least 6."
            )

        return self.matrix[
            3:6,
            3:6,
        ].copy()

    # ==================================================================
    # Acceleration Covariance
    # ==================================================================

    @property
    def acceleration(self) -> np.ndarray:
        """
        3x3 acceleration covariance.

        State indices:

            ax -> 6
            ay -> 7
            az -> 8
        """

        if self.dimension < 9:
            raise ValueError(
                "Covariance dimension must be at least 9."
            )

        return self.matrix[
            6:9,
            6:9,
        ].copy()

    # ==================================================================
    # Position / Velocity Cross Covariance
    # ==================================================================

    @property
    def position_velocity(self) -> np.ndarray:

        if self.dimension < 6:
            raise ValueError(
                "Covariance dimension must be at least 6."
            )

        return self.matrix[
            0:3,
            3:6,
        ].copy()

    # ==================================================================
    # Position / Acceleration Cross Covariance
    # ==================================================================

    @property
    def position_acceleration(self) -> np.ndarray:

        if self.dimension < 9:
            raise ValueError(
                "Covariance dimension must be at least 9."
            )

        return self.matrix[
            0:3,
            6:9,
        ].copy()

    # ==================================================================
    # Velocity / Acceleration Cross Covariance
    # ==================================================================

    @property
    def velocity_acceleration(self) -> np.ndarray:

        if self.dimension < 9:
            raise ValueError(
                "Covariance dimension must be at least 9."
            )

        return self.matrix[
            3:6,
            6:9,
        ].copy()

    # ==================================================================
    # Diagonal
    # ==================================================================

    @property
    def variance(self) -> np.ndarray:
        """
        Variance of every state component.
        """

        return np.diag(
            self.matrix
        ).copy()

    # ==================================================================
    # Standard Deviation
    # ==================================================================

    @property
    def standard_deviation(self) -> np.ndarray:
        """
        Standard deviation of every state component.
        """

        diagonal = np.diag(
            self.matrix
        )

        return np.sqrt(
            np.maximum(
                diagonal,
                0.0,
            )
        )

    # ==================================================================
    # Position Standard Deviation
    # ==================================================================

    @property
    def position_std(self) -> np.ndarray:

        return self.standard_deviation[
            0:3
        ]

    # ==================================================================
    # Velocity Standard Deviation
    # ==================================================================

    @property
    def velocity_std(self) -> np.ndarray:

        if self.dimension < 6:
            raise ValueError(
                "Covariance dimension must be at least 6."
            )

        return self.standard_deviation[
            3:6
        ]

    # ==================================================================
    # Acceleration Standard Deviation
    # ==================================================================

    @property
    def acceleration_std(self) -> np.ndarray:

        if self.dimension < 9:
            raise ValueError(
                "Covariance dimension must be at least 9."
            )

        return self.standard_deviation[
            6:9
        ]

    # ==================================================================
    # Validation
    # ==================================================================

    def is_symmetric(
        self,
        tolerance: float = 1e-9,
    ) -> bool:

        return bool(
            np.allclose(
                self.matrix,
                self.matrix.T,
                atol=tolerance,
            )
        )

    # ==================================================================

    def is_positive_semidefinite(
        self,
        tolerance: float = 1e-9,
    ) -> bool:

        eigenvalues = np.linalg.eigvalsh(
            self.matrix
        )

        return bool(
            np.all(
                eigenvalues >= -tolerance
            )
        )

    # ==================================================================

    def validate(
        self,
        tolerance: float = 1e-9,
    ) -> None:
        """
        Validate covariance for tracking use.
        """

        if not self.is_symmetric(
            tolerance
        ):
            raise ValueError(
                "Covariance matrix must be symmetric."
            )

        if not self.is_positive_semidefinite(
            tolerance
        ):
            raise ValueError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==================================================================
    # Copy
    # ==================================================================

    def copy(self) -> "Covariance":

        return Covariance(
            matrix=self.matrix.copy()
        )

    # ==================================================================
    # Regularization
    # ==================================================================

    def regularize(
        self,
        epsilon: float = 1e-9,
    ) -> "Covariance":
        """
        Add a small diagonal value to improve numerical stability.
        """

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        regularized = (
            self.matrix
            + np.eye(self.dimension)
            * epsilon
        )

        return Covariance(
            matrix=regularized
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def as_dict(self) -> dict:

        return {
            "dimension": self.dimension,
            "shape": self.shape,
            "matrix": self.matrix.tolist(),
            "variance": self.variance.tolist(),
            "standard_deviation":
                self.standard_deviation.tolist(),
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            f"Covariance("
            f"dimension={self.dimension}, "
            f"shape={self.shape})"
        )


# ======================================================================
# Factory Functions
# ======================================================================

def zero_covariance(
    dimension: int = 9,
) -> Covariance:
    """
    Create a zero covariance matrix.

    Normally used only for initialization/testing.
    """

    if dimension <= 0:
        raise ValueError(
            "dimension must be greater than zero."
        )

    return Covariance(
        matrix=np.zeros(
            (dimension, dimension),
            dtype=float,
        )
    )


# ======================================================================

def identity_covariance(
    dimension: int = 9,
    variance: float = 1.0,
) -> Covariance:
    """
    Create an identity covariance matrix.

    Parameters
    ----------
    dimension:
        State dimension.

    variance:
        Diagonal variance.
    """

    if dimension <= 0:
        raise ValueError(
            "dimension must be greater than zero."
        )

    if variance < 0:
        raise ValueError(
            "variance must be non-negative."
        )

    return Covariance(
        matrix=np.eye(
            dimension,
            dtype=float,
        ) * variance
    )


# ======================================================================

def ca_covariance(
    position_variance: float = 100.0,
    velocity_variance: float = 25.0,
    acceleration_variance: float = 4.0,
) -> Covariance:
    """
    Create a default 9x9 covariance for the 3D
    Constant Acceleration model.

    State ordering:

        [x, y, z,
         vx, vy, vz,
         ax, ay, az]
    """

    if position_variance < 0:
        raise ValueError(
            "position_variance must be non-negative."
        )

    if velocity_variance < 0:
        raise ValueError(
            "velocity_variance must be non-negative."
        )

    if acceleration_variance < 0:
        raise ValueError(
            "acceleration_variance must be non-negative."
        )

    matrix = np.zeros(
        (9, 9),
        dtype=float,
    )

    # Position
    matrix[
        0:3,
        0:3,
    ] = np.eye(3) * position_variance

    # Velocity
    matrix[
        3:6,
        3:6,
    ] = np.eye(3) * velocity_variance

    # Acceleration
    matrix[
        6:9,
        6:9,
    ] = np.eye(3) * acceleration_variance

    return Covariance(
        matrix=matrix
    )