"""
========================================================================
Tracking Filter Configuration
========================================================================

Central configuration for all tracking filters.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FilterConfig:
    """
    Configuration for a Constant Velocity (CV) Kalman Filter.
    """

    ####################################################################
    # Initial covariance
    ####################################################################

    initial_position_variance: float = 25.0
    initial_velocity_variance: float = 100.0

    ####################################################################
    # Process noise
    ####################################################################

    process_noise_position: float = 1.0
    process_noise_velocity: float = 5.0

    ####################################################################
    # Measurement noise
    ####################################################################

    measurement_noise_position: float = 4.0

    ####################################################################
    # Helpers
    ####################################################################

    @property
    def initial_covariance(self) -> np.ndarray:

        return np.diag([

            self.initial_position_variance,
            self.initial_position_variance,
            self.initial_position_variance,

            self.initial_velocity_variance,
            self.initial_velocity_variance,
            self.initial_velocity_variance,

        ])

    @property
    def measurement_covariance(self) -> np.ndarray:

        return np.diag([

            self.measurement_noise_position,
            self.measurement_noise_position,
            self.measurement_noise_position,

        ])