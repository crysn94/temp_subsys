from dataclasses import dataclass

from .base import Geometry


@dataclass(frozen=True, slots=True)
class Size2D(Geometry):

    width: float

    height: float

    @property
    def area(self):

        return self.width * self.height

    @property
    def aspect_ratio(self):

        if self.height == 0:

            return 0.0

        return self.width / self.height

@dataclass(frozen=True, slots=True)
class Size3D(Geometry):

    width: float

    height: float

    depth: float

    @property
    def volume(self):

        return (

            self.width *

            self.height *

            self.depth

        )