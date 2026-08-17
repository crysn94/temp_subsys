"""
====================================================================
Geometry Factory
====================================================================

Reusable geometry objects for unit testing.

====================================================================
"""

from random import uniform

from core.geometry import (
    Point2D,
    Point3D,
    Velocity3D,
    BoundingBox2D,
    BoundingBox3D,
)


class GeometryFactory:

    ##################################################################
    # Point2D
    ##################################################################

    @staticmethod
    def point2d(
        x: float = 100.0,
        y: float = 200.0,
    ) -> Point2D:

        return Point2D(x=x, y=y)

    ##################################################################
    # Point3D
    ##################################################################

    @staticmethod
    def point3d(
        x: float = 0.0,
        y: float = 0.0,
        z: float = 100.0,
    ) -> Point3D:

        return Point3D(
            x=x,
            y=y,
            z=z,
        )

    ##################################################################
    # Velocity3D
    ##################################################################

    @staticmethod
    def velocity(
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
    ) -> Velocity3D:

        return Velocity3D(
            x=vx,
            y=vy,
            z=vz,
        )

    ##################################################################
    # BoundingBox2D
    ##################################################################

    @staticmethod
    def bbox2d(
        x: float = 100,
        y: float = 100,
        width: float = 80,
        height: float = 60,
    ) -> BoundingBox2D:

        return BoundingBox2D(
            x=x,
            y=y,
            width=width,
            height=height,
        )

    ##################################################################
    # BoundingBox3D
    ##################################################################

    @staticmethod
    def bbox3d() -> BoundingBox3D:

        return BoundingBox3D(
            center=GeometryFactory.point3d(),
            width=2.0,
            height=1.5,
            depth=1.0,
        )

    ##################################################################
    # Random Point
    ##################################################################

    @staticmethod
    def random_point2d():

        return Point2D(

            x=uniform(0, 640),

            y=uniform(0, 640),

        )

    ##################################################################
    # Random Bounding Box
    ##################################################################

    @staticmethod
    def random_bbox2d():

        return BoundingBox2D(

            x=uniform(0, 500),

            y=uniform(0, 500),

            width=uniform(20, 150),

            height=uniform(20, 150),

        )