"""
========================================================================
Quaternion Geometry
========================================================================

Canonical quaternion representation used throughout the C-UAS framework.

Used by

• Pose3D
• Sensor Fusion
• Drone Attitude
• Camera Orientation
• Radar Alignment
• Gimbal Control
• 3D Tracking

Quaternion format

    q = w + xi + yj + zk

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from math import asin
from math import cos
from math import degrees
from math import radians
from math import sin
from math import sqrt

from .base import Geometry
from .orientation import Orientation


@dataclass(frozen=True, slots=True)
class Quaternion(Geometry):

    w: float
    x: float
    y: float
    z: float

    ####################################################################
    # Constructors
    ####################################################################

    @classmethod
    def identity(cls):
        return cls(1.0, 0.0, 0.0, 0.0)

    ####################################################################
    # Properties
    ####################################################################

    @property
    def magnitude(self) -> float:
        return sqrt(
            self.w**2 +
            self.x**2 +
            self.y**2 +
            self.z**2
        )

    ####################################################################
    # Basic Operations
    ####################################################################

    def normalize(self):

        mag = self.magnitude

        if mag == 0:
            return Quaternion.identity()

        return Quaternion(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag,
        )

    def conjugate(self):

        return Quaternion(
            self.w,
            -self.x,
            -self.y,
            -self.z,
        )

    def inverse(self):

        mag2 = self.magnitude ** 2

        if mag2 == 0:
            return Quaternion.identity()

        q = self.conjugate()

        return Quaternion(
            q.w / mag2,
            q.x / mag2,
            q.y / mag2,
            q.z / mag2,
        )

    ####################################################################
    # Quaternion Multiplication
    ####################################################################

    def multiply(self, other: "Quaternion"):

        return Quaternion(

            self.w * other.w
            - self.x * other.x
            - self.y * other.y
            - self.z * other.z,

            self.w * other.x
            + self.x * other.w
            + self.y * other.z
            - self.z * other.y,

            self.w * other.y
            - self.x * other.z
            + self.y * other.w
            + self.z * other.x,

            self.w * other.z
            + self.x * other.y
            - self.y * other.x
            + self.z * other.w,
        )

    ####################################################################
    # Euler Conversion
    ####################################################################

    @classmethod
    def from_orientation(
        cls,
        orientation: Orientation,
    ):

        roll = radians(orientation.roll)
        pitch = radians(orientation.pitch)
        yaw = radians(orientation.yaw)

        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)

        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)

        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)

        return cls(

            cr * cp * cy + sr * sp * sy,

            sr * cp * cy - cr * sp * sy,

            cr * sp * cy + sr * cp * sy,

            cr * cp * sy - sr * sp * cy,

        )

    ####################################################################

    def to_orientation(self):

        q = self.normalize()

        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)

        roll = atan2(sinr, cosr)

        sinp = 2 * (q.w * q.y - q.z * q.x)

        if abs(sinp) >= 1:
            pitch = (
                3.141592653589793 / 2
                if sinp > 0
                else
                -3.141592653589793 / 2
            )
        else:
            pitch = asin(sinp)

        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)

        yaw = atan2(siny, cosy)

        return Orientation(
            degrees(roll),
            degrees(pitch),
            degrees(yaw),
        )

    ####################################################################
    # Rotation Matrix
    ####################################################################

    def to_rotation_matrix(self):

        q = self.normalize()

        return (

            (
                1 - 2 * (q.y*q.y + q.z*q.z),
                2 * (q.x*q.y - q.z*q.w),
                2 * (q.x*q.z + q.y*q.w),
            ),

            (
                2 * (q.x*q.y + q.z*q.w),
                1 - 2 * (q.x*q.x + q.z*q.z),
                2 * (q.y*q.z - q.x*q.w),
            ),

            (
                2 * (q.x*q.z - q.y*q.w),
                2 * (q.y*q.z + q.x*q.w),
                1 - 2 * (q.x*q.x + q.y*q.y),
            ),

        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_tuple(self):

        return (
            self.w,
            self.x,
            self.y,
            self.z,
        )

    def as_dict(self):

        return {
            "w": self.w,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    def __str__(self):

        return (
            f"Quaternion("
            f"w={self.w:.4f}, "
            f"x={self.x:.4f}, "
            f"y={self.y:.4f}, "
            f"z={self.z:.4f})"
        )