"""
Tracking mathematical utilities.
"""

from .kalman_matrices import (
    STATE_DIMENSION,
    POSITION_DIMENSION,
    MEASUREMENT_DIMENSION,
    constant_velocity_transition_matrix,
    constant_velocity_process_noise,
    position_measurement_matrix,
    position_measurement_noise,
    state_identity_matrix,
    initial_state_covariance,
    validate_state_matrix,
    validate_measurement_matrix,
)

__all__ = [
    "STATE_DIMENSION",
    "POSITION_DIMENSION",
    "MEASUREMENT_DIMENSION",
    "constant_velocity_transition_matrix",
    "constant_velocity_process_noise",
    "position_measurement_matrix",
    "position_measurement_noise",
    "state_identity_matrix",
    "initial_state_covariance",
    "validate_state_matrix",
    "validate_measurement_matrix",
]