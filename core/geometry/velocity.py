"""
========================================================================
Velocity Geometry
========================================================================

Canonical velocity vectors used throughout the C-UAS framework.

Used by

• Tracking
• Radar
• Sensor Fusion
• Threat Assessment
• Trajectory Prediction

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
# Velocity2D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Velocity2D(Geometry):

    vx: float
    vy: float

    ##################################################################

    @property
    def magnitude(self) -> float:
        return hypot(self.vx, self.vy)

    @property
    def speed(self) -> float:
        return self.magnitude

    @property
    def heading(self) -> float:
        """
        Heading in degrees.

        0° = +X axis
        90° = +Y axis
        """

        return degrees(
            atan2(
                self.vy,
                self.vx,
            )
        )

    ##################################################################

    def normalize(self):

        mag = self.magnitude

        if mag == 0:
            return Velocity2D(0.0, 0.0)

        return Velocity2D(

            self.vx / mag,

            self.vy / mag,

        )

    ##################################################################

    def scale(self, factor: float):

        return Velocity2D(

            self.vx * factor,

            self.vy * factor,

        )

    ##################################################################

    def rotate(self, angle_deg: float):

        theta = radians(angle_deg)

        c = cos(theta)

        s = sin(theta)

        return Velocity2D(

            self.vx * c - self.vy * s,

            self.vx * s + self.vy * c,

        )

    ##################################################################

    def dot(self, other: "Velocity2D"):

        return (

            self.vx * other.vx +

            self.vy * other.vy

        )

    ##################################################################

    def as_tuple(self):

        return (

            self.vx,

            self.vy,

        )


# ======================================================================
# Velocity3D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Velocity3D(Geometry):

    vx: float
    vy: float
    vz: float

    ##################################################################

    @property
    def magnitude(self):

        return sqrt(

            self.vx ** 2 +

            self.vy ** 2 +

            self.vz ** 2

        )

    @property
    def speed(self):

        return self.magnitude

    ##################################################################

    def normalize(self):

        mag = self.magnitude

        if mag == 0:

            return Velocity3D(

                0.0,

                0.0,

                0.0,

            )

        return Velocity3D(

            self.vx / mag,

            self.vy / mag,

            self.vz / mag,

        )

    ##################################################################

    def scale(self, factor):

        return Velocity3D(

            self.vx * factor,

            self.vy * factor,

            self.vz * factor,

        )

    ##################################################################

    def dot(self, other: "Velocity3D"):

        return (

            self.vx * other.vx +

            self.vy * other.vy +

            self.vz * other.vz

        )

    ##################################################################

    def cross(self, other: "Velocity3D"):

        return Velocity3D(

            self.vy * other.vz - self.vz * other.vy,

            self.vz * other.vx - self.vx * other.vz,

            self.vx * other.vy - self.vy * other.vx,

        )

    ##################################################################

    def as_tuple(self):

        return (

            self.vx,

            self.vy,

            self.vz,

        )

    @property
    def x(self) -> float:
        return self.vx

    @property
    def y(self) -> float:
        return self.vy

    @property
    def z(self) -> float:
        return self.vz