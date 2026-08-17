"""
========================================================================
Measurement Uncertainty
========================================================================

Represents the uncertainty associated with a measurement.

Used by

• Detection
• Tracking
• Radar
• EO
• RF
• Sensor Fusion

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass(slots=True, frozen=True)
class MeasurementUncertainty:

    ####################################################################
    # Position Standard Deviations
    ####################################################################

    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_z: float = 0.0

    ####################################################################
    # Velocity Standard Deviations
    ####################################################################

    sigma_vx: float = 0.0
    sigma_vy: float = 0.0
    sigma_vz: float = 0.0

    ####################################################################
    # Orientation Standard Deviations
    ####################################################################

    sigma_roll: float = 0.0
    sigma_pitch: float = 0.0
    sigma_yaw: float = 0.0

    ####################################################################
    # Image Space
    ####################################################################

    sigma_width: float = 0.0
    sigma_height: float = 0.0

    ####################################################################
    # Optional Covariance Matrix
    ####################################################################

    covariance: list[list[float]] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    ####################################################################
    # Validation
    ####################################################################

    def __post_init__(self):

        values = [

            self.sigma_x,
            self.sigma_y,
            self.sigma_z,

            self.sigma_vx,
            self.sigma_vy,
            self.sigma_vz,

            self.sigma_roll,
            self.sigma_pitch,
            self.sigma_yaw,

            self.sigma_width,
            self.sigma_height,

        ]

        if any(v < 0 for v in values):
            raise ValueError(
                "Standard deviations cannot be negative."
            )

    ####################################################################
    # Derived Quantities
    ####################################################################

    @property
    def position_rms(self) -> float:

        return sqrt(

            self.sigma_x**2 +

            self.sigma_y**2 +

            self.sigma_z**2

        )

    @property
    def velocity_rms(self) -> float:

        return sqrt(

            self.sigma_vx**2 +

            self.sigma_vy**2 +

            self.sigma_vz**2

        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "position": {

                "x": self.sigma_x,

                "y": self.sigma_y,

                "z": self.sigma_z,

            },

            "velocity": {

                "vx": self.sigma_vx,

                "vy": self.sigma_vy,

                "vz": self.sigma_vz,

            },

            "orientation": {

                "roll": self.sigma_roll,

                "pitch": self.sigma_pitch,

                "yaw": self.sigma_yaw,

            },

            "bbox": {

                "width": self.sigma_width,

                "height": self.sigma_height,

            },

            "covariance": self.covariance,

            "metadata": self.metadata,

        }

    ####################################################################

    def __str__(self):

        return (

            "MeasurementUncertainty("

            f"Position RMS={self.position_rms:.3f}, "

            f"Velocity RMS={self.velocity_rms:.3f}"

            ")"

        )