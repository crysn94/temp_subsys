"""
========================================================================
Orientation Geometry
========================================================================

Canonical orientation representation for the C-UAS framework.

Represents orientation using intrinsic Roll-Pitch-Yaw (XYZ) Euler angles.

Used by

• EO/IR Cameras
• Gimbals
• Radar
• UAVs
• Tracking
• Sensor Fusion
• Pose3D

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from math import degrees
from math import sin
from math import cos

from .base import Geometry


# ======================================================================
# Orientation
# ======================================================================

@dataclass(frozen=True, slots=True)
class Orientation(Geometry):
    """
    Roll, Pitch, Yaw orientation in degrees.
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    ####################################################################
    # Helpers
    ####################################################################

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-180, 180).
        """
        angle = (angle + 180.0) % 360.0 - 180.0
        return angle

    ####################################################################
    # Properties
    ####################################################################

    @property
    def heading(self) -> float:
        """
        Heading equals yaw.
        """
        return self.yaw

    @property
    def elevation(self) -> float:
        """
        Elevation equals pitch.
        """
        return self.pitch

    ####################################################################
    # Conversions
    ####################################################################

    def as_tuple(self):

        return (
            self.roll,
            self.pitch,
            self.yaw,
        )

    def as_radians(self):

        return (
            radians(self.roll),
            radians(self.pitch),
            radians(self.yaw),
        )

    ####################################################################
    # Normalization
    ####################################################################

    def normalize(self):

        return Orientation(
            self._normalize_angle(self.roll),
            self._normalize_angle(self.pitch),
            self._normalize_angle(self.yaw),
        )

    ####################################################################
    # Rotation Matrix
    ####################################################################

    def rotation_matrix(self):
        """
        Returns a 3x3 rotation matrix (ZYX convention).

        Matrix returned as tuple-of-tuples.
        """

        roll, pitch, yaw = self.as_radians()

        cr = cos(roll)
        sr = sin(roll)

        cp = cos(pitch)
        sp = sin(pitch)

        cy = cos(yaw)
        sy = sin(yaw)

        return (
            (
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ),
            (
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ),
            (
                -sp,
                cp * sr,
                cp * cr,
            ),
        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }

    ####################################################################
    # Representation
    ####################################################################

    def __str__(self):

        return (
            f"Orientation("
            f"roll={self.roll:.2f}, "
            f"pitch={self.pitch:.2f}, "
            f"yaw={self.yaw:.2f})"
        )