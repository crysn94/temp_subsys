"""
========================================================================
Geometry Transformations
========================================================================

Common geometric transformations used throughout the C-UAS framework.

Supports

• Translation
• Scaling
• Rotation
• Coordinate conversion
• Image ↔ World transforms

========================================================================
"""

from __future__ import annotations

from math import cos
from math import radians
from math import sin

from .bbox import BoundingBox2D
from .point import Point2D
from .polygon import Polygon2D


class GeometryTransform:
    """
    Static transformation utilities.
    """

    # ------------------------------------------------------------------
    # Point
    # ------------------------------------------------------------------

    @staticmethod
    def translate_point(
        point: Point2D,
        dx: float,
        dy: float,
    ) -> Point2D:

        return Point2D(
            point.x + dx,
            point.y + dy,
        )

    @staticmethod
    def scale_point(
        point: Point2D,
        sx: float,
        sy: float,
    ) -> Point2D:

        return Point2D(
            point.x * sx,
            point.y * sy,
        )

    @staticmethod
    def rotate_point(
        point: Point2D,
        angle_deg: float,
        origin: Point2D = Point2D(0.0, 0.0),
    ) -> Point2D:

        theta = radians(angle_deg)

        c = cos(theta)
        s = sin(theta)

        x = point.x - origin.x
        y = point.y - origin.y

        return Point2D(
            x * c - y * s + origin.x,
            x * s + y * c + origin.y,
        )

    # ------------------------------------------------------------------
    # Bounding Box
    # ------------------------------------------------------------------

    @staticmethod
    def translate_bbox(
        bbox: BoundingBox2D,
        dx: float,
        dy: float,
    ) -> BoundingBox2D:

        return bbox.translate(dx, dy)

    @staticmethod
    def scale_bbox(
        bbox: BoundingBox2D,
        sx: float,
        sy: float,
    ) -> BoundingBox2D:

        return bbox.scale(sx, sy)

    # ------------------------------------------------------------------
    # Polygon
    # ------------------------------------------------------------------

    @staticmethod
    def translate_polygon(
        polygon: Polygon2D,
        dx: float,
        dy: float,
    ) -> Polygon2D:

        return polygon.translate(dx, dy)

    @staticmethod
    def scale_polygon(
        polygon: Polygon2D,
        sx: float,
        sy: float,
    ) -> Polygon2D:

        return polygon.scale(sx, sy)

    @staticmethod
    def rotate_polygon(
        polygon: Polygon2D,
        angle_deg: float,
    ) -> Polygon2D:

        return polygon.rotate(angle_deg)