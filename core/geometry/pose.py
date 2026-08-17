"""
========================================================================
Pose Geometry
========================================================================

Canonical pose representation used throughout the C-UAS framework.

Pose = Position + Orientation + Motion

Used by

• EO Sensors
• IR Sensors
• Radar
• RF Sensors
• Acoustic Arrays
• Sensor Fusion
• Tracking
• Navigation
• Threat Assessment

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from math import sqrt

from .acceleration import Acceleration2D
from .acceleration import Acceleration3D
from .orientation import Orientation
from .point import Point2D
from .point import Point3D
from .quaternion import Quaternion
from .velocity import Velocity2D
from .velocity import Velocity3D


# ======================================================================
# Pose2D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Pose2D:

    position: Point2D

    heading: float = 0.0

    velocity: Velocity2D | None = None

    acceleration: Acceleration2D | None = None

    ##################################################################

    def distance_to(
        self,
        other: "Pose2D",
    ) -> float:

        dx = self.position.x - other.position.x

        dy = self.position.y - other.position.y

        return hypot(dx, dy)

    ##################################################################

    def translate(
        self,
        dx: float,
        dy: float,
    ):

        return Pose2D(

            position=Point2D(
                self.position.x + dx,
                self.position.y + dy,
            ),

            heading=self.heading,

            velocity=self.velocity,

            acceleration=self.acceleration,

        )

    ##################################################################

    def rotate(
        self,
        heading: float,
    ):

        return Pose2D(

            position=self.position,

            heading=self.heading + heading,

            velocity=self.velocity,

            acceleration=self.acceleration,

        )

    ##################################################################

    def as_dict(self):

        return {

            "position": self.position.as_dict(),

            "heading": self.heading,

            "velocity": (
                self.velocity.as_tuple()
                if self.velocity else None
            ),

            "acceleration": (
                self.acceleration.as_tuple()
                if self.acceleration else None
            ),

        }


# ======================================================================
# Pose3D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Pose3D:

    position: Point3D

    orientation: Quaternion

    velocity: Velocity3D | None = None

    acceleration: Acceleration3D | None = None

    ##################################################################

    @classmethod
    def from_orientation(

        cls,

        position: Point3D,

        orientation: Orientation,

        velocity: Velocity3D | None = None,

        acceleration: Acceleration3D | None = None,

    ):

        return cls(

            position=position,

            orientation=Quaternion.from_orientation(
                orientation
            ),

            velocity=velocity,

            acceleration=acceleration,

        )

    ##################################################################

    def distance_to(

        self,

        other: "Pose3D",

    ):

        dx = self.position.x - other.position.x

        dy = self.position.y - other.position.y

        dz = self.position.z - other.position.z

        return sqrt(

            dx * dx +

            dy * dy +

            dz * dz

        )

    ##################################################################

    def copy(

        self,

        **kwargs,

    ):

        data = {

            "position": self.position,

            "orientation": self.orientation,

            "velocity": self.velocity,

            "acceleration": self.acceleration,

        }

        data.update(kwargs)

        return Pose3D(**data)

    ##################################################################

    def as_dict(self):

        return {

            "position": self.position.as_dict(),

            "orientation": self.orientation.as_dict(),

            "velocity": (
                self.velocity.as_tuple()
                if self.velocity else None
            ),

            "acceleration": (
                self.acceleration.as_tuple()
                if self.acceleration else None
            ),

        }

    ##################################################################

    def as_tuple(self):

        return (

            self.position,

            self.orientation,

            self.velocity,

            self.acceleration,

        )