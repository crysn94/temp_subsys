"""
========================================================================
Kalman Filter Matrix Utilities
========================================================================

Matrix construction utilities for the C-UAS tracking framework.

Primary model
-------------

3D Constant Velocity (CV):

    X = [x, y, z, vx, vy, vz]^T

Supported operations
--------------------

• State transition matrix F
• Process noise matrix Q
• Position measurement matrix H
• Measurement noise matrix R
• Identity matrix

These utilities are intentionally independent of:

• Track
• DetectionResult
• Sensor implementations
• KalmanFilter implementation

They can therefore be reused by:

• Kalman Filter
• Extended Kalman Filter
• IMM
• Sensor Fusion
• Prediction
• Simulation

========================================================================
"""

from __future__ import annotations

import numpy as np


# ======================================================================
# Constants
# ======================================================================

STATE_DIMENSION = 6
POSITION_DIMENSION = 3
MEASUREMENT_DIMENSION = 3


# ======================================================================
# Validation
# ======================================================================

def _validate_dt(dt: float) -> float:
    """
    Validate and normalize the time step.

    Parameters
    ----------
    dt:
        Time step in seconds.

    Returns
    -------
    float
        Validated time step.
    """

    dt = float(dt)

    if not np.isfinite(dt):
        raise ValueError(
            "dt must be finite."
        )

    if dt < 0.0:
        raise ValueError(
            "dt cannot be negative."
        )

    return dt


def _validate_non_negative(
    value: float,
    name: str,
) -> float:
    """
    Validate a non-negative scalar.
    """

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if value < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


# ======================================================================
# State Transition Matrix
# ======================================================================

def constant_velocity_transition_matrix(
    dt: float,
) -> np.ndarray:
    """
    Construct the 3D Constant Velocity state transition matrix.

    State:

        [x, y, z, vx, vy, vz]

    Model:

        x(k+1)  = x(k) + vx(k) * dt
        y(k+1)  = y(k) + vy(k) * dt
        z(k+1)  = z(k) + vz(k) * dt

        vx(k+1) = vx(k)
        vy(k+1) = vy(k)
        vz(k+1) = vz(k)

    Parameters
    ----------
    dt:
        Time step in seconds.

    Returns
    -------
    numpy.ndarray
        6x6 transition matrix.
    """

    dt = _validate_dt(dt)

    return np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],

            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


# ======================================================================
# Process Noise
# ======================================================================

def constant_velocity_process_noise(
    dt: float,
    acceleration_variance: float = 1.0,
) -> np.ndarray:
    """
    Construct the process noise covariance matrix for a 3D
    Constant Velocity model driven by white acceleration noise.

    For one axis:

        Q_axis = q *
                 [[dt^4 / 4, dt^3 / 2],
                  [dt^3 / 2, dt^2]]

    The complete 3D matrix is constructed for:

        x, y, z, vx, vy, vz

    Parameters
    ----------
    dt:
        Time step in seconds.

    acceleration_variance:
        Variance of unmodeled acceleration.

    Returns
    -------
    numpy.ndarray
        6x6 process noise covariance matrix.
    """

    dt = _validate_dt(dt)

    acceleration_variance = _validate_non_negative(
        acceleration_variance,
        "acceleration_variance",
    )

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    q_position = acceleration_variance * dt4 / 4.0
    q_cross = acceleration_variance * dt3 / 2.0
    q_velocity = acceleration_variance * dt2

    return np.array(
        [
            [q_position, 0.0, 0.0, q_cross, 0.0, 0.0],
            [0.0, q_position, 0.0, 0.0, q_cross, 0.0],
            [0.0, 0.0, q_position, 0.0, 0.0, q_cross],

            [q_cross, 0.0, 0.0, q_velocity, 0.0, 0.0],
            [0.0, q_cross, 0.0, 0.0, q_velocity, 0.0],
            [0.0, 0.0, q_cross, 0.0, 0.0, q_velocity],
        ],
        dtype=float,
    )


# ======================================================================
# Position Measurement Matrix
# ======================================================================

def position_measurement_matrix() -> np.ndarray:
    """
    Construct the observation matrix for position measurements.

    Measurement:

        Z = [x, y, z]

    State:

        X = [x, y, z, vx, vy, vz]

    Returns
    -------
    numpy.ndarray
        3x6 observation matrix.
    """

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


# ======================================================================
# Measurement Noise
# ======================================================================

def position_measurement_noise(
    position_variance: float = 4.0,
) -> np.ndarray:
    """
    Construct measurement noise covariance for 3D position.

    Parameters
    ----------
    position_variance:
        Position measurement variance.

    Returns
    -------
    numpy.ndarray
        3x3 measurement covariance matrix.
    """

    position_variance = _validate_non_negative(
        position_variance,
        "position_variance",
    )

    return np.diag(
        [
            position_variance,
            position_variance,
            position_variance,
        ]
    ).astype(float)


# ======================================================================
# Identity Matrix
# ======================================================================

def state_identity_matrix() -> np.ndarray:
    """
    Return the 6x6 identity matrix.
    """

    return np.eye(
        STATE_DIMENSION,
        dtype=float,
    )


# ======================================================================
# Initial Covariance
# ======================================================================

def initial_state_covariance(
    position_variance: float = 25.0,
    velocity_variance: float = 100.0,
) -> np.ndarray:
    """
    Construct initial covariance for a 3D CV state.

    State ordering:

        x, y, z, vx, vy, vz

    Parameters
    ----------
    position_variance:
        Initial position variance.

    velocity_variance:
        Initial velocity variance.

    Returns
    -------
    numpy.ndarray
        6x6 covariance matrix.
    """

    position_variance = _validate_non_negative(
        position_variance,
        "position_variance",
    )

    velocity_variance = _validate_non_negative(
        velocity_variance,
        "velocity_variance",
    )

    return np.diag(
        [
            position_variance,
            position_variance,
            position_variance,
            velocity_variance,
            velocity_variance,
            velocity_variance,
        ]
    ).astype(float)


# ======================================================================
# Generic Helpers
# ======================================================================

def validate_state_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Validate a 6x6 state-space matrix.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.shape != (
        STATE_DIMENSION,
        STATE_DIMENSION,
    ):
        raise ValueError(
            "State matrix must have shape (6, 6)."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "State matrix contains non-finite values."
        )

    return matrix


def validate_measurement_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Validate a 3x6 measurement matrix.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.shape != (
        MEASUREMENT_DIMENSION,
        STATE_DIMENSION,
    ):
        raise ValueError(
            "Measurement matrix must have shape (3, 6)."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Measurement matrix contains non-finite values."
        )

    return matrix