"""
========================================================================
Acceleration Geometry
========================================================================

Canonical acceleration vectors used throughout the C-UAS framework.

Used by

• Motion Models
• Kalman Filters
• IMM
• Trajectory Prediction
• Threat Assessment
• Guidance Algorithms

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from math import cos
from math import degrees
from math import hypot
from math import radians
from math import sin
from math import sqrt

from .base import Geometry


# ======================================================================
# Acceleration2D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Acceleration2D(Geometry):

    ax: float
    ay: float

    ##################################################################

    @property
    def magnitude(self) -> float:
        return hypot(self.ax, self.ay)

    @property
    def direction(self) -> float:
        """
        Direction in degrees.
        """
        return degrees(
            atan2(
                self.ay,
                self.ax,
            )
        )

    ##################################################################

    def normalize(self):

        mag = self.magnitude

        if mag == 0:
            return Acceleration2D(0.0, 0.0)

        return Acceleration2D(
            self.ax / mag,
            self.ay / mag,
        )

    ##################################################################

    def scale(self, factor: float):

        return Acceleration2D(
            self.ax * factor,
            self.ay * factor,
        )

    ##################################################################

    def rotate(self, angle_deg: float):

        theta = radians(angle_deg)

        c = cos(theta)
        s = sin(theta)

        return Acceleration2D(
            self.ax * c - self.ay * s,
            self.ax * s + self.ay * c,
        )

    ##################################################################

    def dot(self, other: "Acceleration2D"):

        return (
            self.ax * other.ax +
            self.ay * other.ay
        )

    ##################################################################

    def as_tuple(self):

        return (
            self.ax,
            self.ay,
        )


# ======================================================================
# Acceleration3D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Acceleration3D(Geometry):

    ax: float
    ay: float
    az: float

    ##################################################################

    @property
    def magnitude(self):

        return sqrt(
            self.ax ** 2 +
            self.ay ** 2 +
            self.az ** 2
        )

    ##################################################################

    def normalize(self):

        mag = self.magnitude

        if mag == 0:
            return Acceleration3D(
                0.0,
                0.0,
                0.0,
            )

        return Acceleration3D(
            self.ax / mag,
            self.ay / mag,
            self.az / mag,
        )

    ##################################################################

    def scale(self, factor):

        return Acceleration3D(
            self.ax * factor,
            self.ay * factor,
            self.az * factor,
        )

    ##################################################################

    def dot(self, other: "Acceleration3D"):

        return (
            self.ax * other.ax +
            self.ay * other.ay +
            self.az * other.az
        )

    ##################################################################

    def cross(self, other: "Acceleration3D"):

        return Acceleration3D(
            self.ay * other.az - self.az * other.ay,
            self.az * other.ax - self.ax * other.az,
            self.ax * other.ay - self.ay * other.ax,
        )

    ##################################################################

    def as_tuple(self):

        return (
            self.ax,
            self.ay,
            self.az,
        )