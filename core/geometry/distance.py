"""
========================================================================
Distance Utilities
========================================================================

Canonical distance computations used throughout the C-UAS framework.

Supports

• Euclidean
• Manhattan
• Chebyshev
• Minkowski
• Cosine
• Mahalanobis
• Point-to-Line
• Point-to-Segment
• Point-to-BoundingBox

========================================================================
"""

from __future__ import annotations

from math import hypot
from math import sqrt

from .point import Point2D, Point3D
from .bbox import BoundingBox2D


class Distance:

    ####################################################################
    # Euclidean
    ####################################################################

    @staticmethod
    def euclidean(
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        return hypot(
            p1.x - p2.x,
            p1.y - p2.y,
        )

    @staticmethod
    def euclidean3d(
        p1: Point3D,
        p2: Point3D,
    ) -> float:

        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z

        return sqrt(dx * dx + dy * dy + dz * dz)

    ####################################################################
    # Squared Distance
    ####################################################################

    @staticmethod
    def squared(
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        return dx * dx + dy * dy

    ####################################################################
    # Manhattan
    ####################################################################

    @staticmethod
    def manhattan(
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        return (
            abs(p1.x - p2.x)
            +
            abs(p1.y - p2.y)
        )

    ####################################################################
    # Chebyshev
    ####################################################################

    @staticmethod
    def chebyshev(
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        return max(
            abs(p1.x - p2.x),
            abs(p1.y - p2.y),
        )

    ####################################################################
    # Minkowski
    ####################################################################

    @staticmethod
    def minkowski(
        p1: Point2D,
        p2: Point2D,
        p: float = 3.0,
    ) -> float:

        return (
            abs(p1.x - p2.x) ** p
            +
            abs(p1.y - p2.y) ** p
        ) ** (1.0 / p)

    ####################################################################
    # Cosine Distance
    ####################################################################

    @staticmethod
    def cosine(
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        dot = p1.x * p2.x + p1.y * p2.y

        n1 = hypot(p1.x, p1.y)
        n2 = hypot(p2.x, p2.y)

        if n1 == 0 or n2 == 0:
            return 1.0

        return 1.0 - dot / (n1 * n2)

    ####################################################################
    # Point to Bounding Box
    ####################################################################

    @staticmethod
    def point_to_bbox(
        point: Point2D,
        bbox: BoundingBox2D,
    ) -> float:

        dx = max(
            bbox.x1 - point.x,
            0,
            point.x - bbox.x2,
        )

        dy = max(
            bbox.y1 - point.y,
            0,
            point.y - bbox.y2,
        )

        return hypot(dx, dy)

    ####################################################################
    # Point to Segment
    ####################################################################

    @staticmethod
    def point_to_segment(
        point: Point2D,
        start: Point2D,
        end: Point2D,
    ) -> float:

        dx = end.x - start.x
        dy = end.y - start.y

        if dx == 0 and dy == 0:
            return Distance.euclidean(point, start)

        t = (
            (
                (point.x - start.x) * dx
                +
                (point.y - start.y) * dy
            )
            /
            (
                dx * dx
                +
                dy * dy
            )
        )

        t = max(0.0, min(1.0, t))

        projection = Point2D(
            start.x + t * dx,
            start.y + t * dy,
        )

        return Distance.euclidean(
            point,
            projection,
        )