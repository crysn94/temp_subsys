"""
========================================================================
Point Geometry
========================================================================

Canonical point primitives used throughout the C-UAS framework.

Provides:

• Point2D
• Point3D

Supports:

• Vector arithmetic
• Distance calculations
• Interpolation
• Bearings
• Elevation
• Serialization

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, sqrt

from .base import Geometry


# ======================================================================
# Point2D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Point2D(Geometry):

    x: float
    y: float

    ####################################################################
    # Basic
    ####################################################################

    def as_tuple(self):
        return (self.x, self.y)

    def as_dict(self):
        return {
            "x": self.x,
            "y": self.y,
        }

    def copy(self, **kwargs):

        return Point2D(
            kwargs.get("x", self.x),
            kwargs.get("y", self.y),
        )

    ####################################################################
    # Vector Math
    ####################################################################

    def __add__(self, other: "Point2D"):

        return Point2D(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: "Point2D"):

        return Point2D(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, scalar: float):

        return Point2D(
            self.x * scalar,
            self.y * scalar,
        )

    def __truediv__(self, scalar: float):

        return Point2D(
            self.x / scalar,
            self.y / scalar,
        )

    ####################################################################
    # Geometry
    ####################################################################

    def distance_to(self, other: "Point2D") -> float:

        return hypot(
            self.x - other.x,
            self.y - other.y,
        )

    def distance_squared(self, other: "Point2D") -> float:

        dx = self.x - other.x
        dy = self.y - other.y

        return dx * dx + dy * dy

    def midpoint(self, other: "Point2D"):

        return Point2D(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
        )

    def bearing_to(self, other: "Point2D"):

        return degrees(

            atan2(

                other.y - self.y,

                other.x - self.x,

            )

        )

    ####################################################################
    # Linear Algebra
    ####################################################################

    def norm(self):

        return hypot(self.x, self.y)

    def normalize(self):

        n = self.norm()

        if n == 0:

            return Point2D(0.0, 0.0)

        return self / n

    def dot(self, other: "Point2D"):

        return (

            self.x * other.x +

            self.y * other.y

        )

    ####################################################################
    # Interpolation
    ####################################################################

    def lerp(self, other: "Point2D", t: float):

        return Point2D(

            self.x + (other.x - self.x) * t,

            self.y + (other.y - self.y) * t,

        )

    ####################################################################

    def __iter__(self):

        yield self.x
        yield self.y


# ======================================================================
# Point3D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Point3D(Geometry):

    x: float
    y: float
    z: float

    ####################################################################
    # Basic
    ####################################################################

    def as_tuple(self):

        return (
            self.x,
            self.y,
            self.z,
        )

    def as_dict(self):

        return {

            "x": self.x,

            "y": self.y,

            "z": self.z,

        }

    ####################################################################
    # Vector Arithmetic
    ####################################################################

    def __add__(self, other: "Point3D"):

        return Point3D(

            self.x + other.x,

            self.y + other.y,

            self.z + other.z,

        )

    def __sub__(self, other: "Point3D"):

        return Point3D(

            self.x - other.x,

            self.y - other.y,

            self.z - other.z,

        )

    def __mul__(self, scalar: float):

        return Point3D(

            self.x * scalar,

            self.y * scalar,

            self.z * scalar,

        )

    def __truediv__(self, scalar: float):

        return Point3D(

            self.x / scalar,

            self.y / scalar,

            self.z / scalar,

        )

    ####################################################################
    # Geometry
    ####################################################################

    def distance_to(self, other: "Point3D"):

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z

        return sqrt(dx * dx + dy * dy + dz * dz)

    def midpoint(self, other: "Point3D"):

        return Point3D(

            (self.x + other.x) / 2,

            (self.y + other.y) / 2,

            (self.z + other.z) / 2,

        )

    ####################################################################
    # Linear Algebra
    ####################################################################

    def norm(self):

        return sqrt(

            self.x ** 2 +

            self.y ** 2 +

            self.z ** 2

        )

    def normalize(self):

        n = self.norm()

        if n == 0:

            return Point3D(0.0, 0.0, 0.0)

        return self / n

    def dot(self, other: "Point3D"):

        return (

            self.x * other.x +

            self.y * other.y +

            self.z * other.z

        )

    def cross(self, other: "Point3D"):

        return Point3D(

            self.y * other.z - self.z * other.y,

            self.z * other.x - self.x * other.z,

            self.x * other.y - self.y * other.x,

        )

    def elevation_to(self, other: "Point3D"):

        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z

        horizontal = hypot(dx, dy)

        return degrees(

            atan2(

                dz,

                horizontal,

            )

        )

    ####################################################################

    def __iter__(self):

        yield self.x
        yield self.y
        yield self.z