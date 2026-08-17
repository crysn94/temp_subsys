"""
========================================================================
2D Polygon Geometry
========================================================================

Canonical polygon representation used throughout the C-UAS framework.

Supports

• YOLO Segmentation
• SAM2
• GroundingDINO
• Sensor Fusion
• Geofencing
• Radar Contours

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import radians
from math import sin

from .base import Geometry
from .bbox import BoundingBox2D
from .point import Point2D


@dataclass(frozen=True, slots=True)
class Polygon2D(Geometry):

    vertices: tuple[Point2D, ...]

    ####################################################################
    # Validation
    ####################################################################

    def __post_init__(self):

        if len(self.vertices) < 3:
            raise ValueError(
                "Polygon requires at least 3 vertices."
            )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def num_vertices(self) -> int:

        return len(self.vertices)

    ####################################################################

    @property
    def bounding_box(self) -> BoundingBox2D:

        xs = [v.x for v in self.vertices]

        ys = [v.y for v in self.vertices]

        return BoundingBox2D.from_xyxy(

            min(xs),

            min(ys),

            max(xs),

            max(ys),

        )

    ####################################################################

    @property
    def perimeter(self) -> float:

        total = 0.0

        for i in range(len(self.vertices)):

            total += self.vertices[i].distance_to(

                self.vertices[(i + 1) % len(self.vertices)]

            )

        return total

    ####################################################################

    @property
    def area(self) -> float:

        area = 0.0

        n = len(self.vertices)

        for i in range(n):

            x1 = self.vertices[i].x

            y1 = self.vertices[i].y

            x2 = self.vertices[(i + 1) % n].x

            y2 = self.vertices[(i + 1) % n].y

            area += x1 * y2

            area -= y1 * x2

        return abs(area) * 0.5

    ####################################################################

    @property
    def centroid(self) -> Point2D:

        cx = sum(v.x for v in self.vertices)

        cy = sum(v.y for v in self.vertices)

        return Point2D(

            cx / len(self.vertices),

            cy / len(self.vertices),

        )

    ####################################################################
    # Geometry
    ####################################################################

    def contains_point(

        self,

        point: Point2D,

    ) -> bool:

        inside = False

        j = len(self.vertices) - 1

        for i in range(len(self.vertices)):

            xi = self.vertices[i].x
            yi = self.vertices[i].y

            xj = self.vertices[j].x
            yj = self.vertices[j].y

            intersect = (

                (yi > point.y) != (yj > point.y)

            ) and (

                point.x

                <

                (xj - xi)

                *

                (point.y - yi)

                /

                (yj - yi + 1e-12)

                +

                xi

            )

            if intersect:

                inside = not inside

            j = i

        return inside

    ####################################################################
    # Transformations
    ####################################################################

    def translate(

        self,

        dx: float,

        dy: float,

    ):

        return Polygon2D(

            tuple(

                Point2D(

                    p.x + dx,

                    p.y + dy,

                )

                for p in self.vertices

            )

        )

    ####################################################################

    def scale(

        self,

        sx: float,

        sy: float,

    ):

        return Polygon2D(

            tuple(

                Point2D(

                    p.x * sx,

                    p.y * sy,

                )

                for p in self.vertices

            )

        )

    ####################################################################

    def rotate(

        self,

        angle_deg: float,

        center: Point2D | None = None,

    ):

        if center is None:

            center = self.centroid

        theta = radians(angle_deg)

        c = cos(theta)

        s = sin(theta)

        points = []

        for p in self.vertices:

            x = p.x - center.x

            y = p.y - center.y

            xr = x * c - y * s

            yr = x * s + y * c

            points.append(

                Point2D(

                    xr + center.x,

                    yr + center.y,

                )

            )

        return Polygon2D(tuple(points))

    ####################################################################
    # Serialization
    ####################################################################

    def to_xy(self):

        return [

            (p.x, p.y)

            for p in self.vertices

        ]

    ####################################################################

    def as_dict(self):

        return {

            "vertices": [

                {

                    "x": p.x,

                    "y": p.y,

                }

                for p in self.vertices

            ]

        }