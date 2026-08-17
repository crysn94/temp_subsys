"""
========================================================================
Bounding Box Geometry
========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Geometry
from .point import Point2D, Point3D
from .size import Size2D

from math import atan, pi

@dataclass(frozen=True, slots=True)
class BoundingBox2D(Geometry):
    """
    Axis-aligned 2D bounding box.

    Coordinate system:

        (x1,y1) ---------
           |             |
           |             |
           |             |
           -------- (x2,y2)
    """

    x1: float
    y1: float
    x2: float
    y2: float

    ####################################################################
    # Constructors
    ####################################################################

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ):
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        return cls(
            x,
            y,
            x + width,
            y + height,
        )

    @classmethod
    def from_center(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: float,
    ):
        return cls(
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        )

    ####################################################################
    # Properties
    ####################################################################

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)

    @property
    def size(self) -> Size2D:
        return Size2D(
            self.width,
            self.height,
        )

    @property
    def center(self) -> Point2D:
        return Point2D(
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def is_valid(self) -> bool:
        return (
            self.x2 >= self.x1
            and self.y2 >= self.y1
        )

    ########################################################################
    # Validation
    ########################################################################

    @property
    def is_empty(self) -> bool:
        return self.area == 0.0

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    @property
    def is_normalized(self) -> bool:
        return (
                0.0 <= self.x1 <= 1.0
                and 0.0 <= self.y1 <= 1.0
                and 0.0 <= self.x2 <= 1.0
                and 0.0 <= self.y2 <= 1.0
        )

    def is_inside(
            self,
            image_width: float,
            image_height: float,
    ) -> bool:

        return (
                self.x1 >= 0
                and self.y1 >= 0
                and self.x2 <= image_width
                and self.y2 <= image_height
        )

    ####################################################################
    # Conversions
    ####################################################################

    def to_xyxy(self):
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

    def to_xywh(self):
        return (
            self.x1,
            self.y1,
            self.width,
            self.height,
        )

    def to_center(self):
        return (
            self.center.x,
            self.center.y,
            self.width,
            self.height,
        )

    ####################################################################
    # Geometry
    ####################################################################

    def contains_point(
        self,
        point: Point2D,
    ) -> bool:

        return (
            self.x1 <= point.x <= self.x2
            and
            self.y1 <= point.y <= self.y2
        )

    def contains_bbox(
        self,
        other: "BoundingBox2D",
    ) -> bool:

        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def intersection(
        self,
        other: "BoundingBox2D",
    ) -> float:

        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)

        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        if x_right <= x_left:
            return 0.0

        if y_bottom <= y_top:
            return 0.0

        return (
            (x_right - x_left)
            *
            (y_bottom - y_top)
        )

    def union(
        self,
        other: "BoundingBox2D",
    ) -> float:

        return (
            self.area
            +
            other.area
            -
            self.intersection(other)
        )

    def iou(
        self,
        other: "BoundingBox2D",
    ) -> float:

        union = self.union(other)

        if union == 0:
            return 0.0

        return self.intersection(other) / union

    ########################################################################
    # Detection Metrics
    ########################################################################

    ####################################################################
    # Intersection over Area (IoA)
    ####################################################################

    def ioa(
            self,
            other: "BoundingBox2D",
    ) -> float:

        if other.area == 0:
            return 0.0

        return self.intersection(other) / other.area

    ####################################################################
    # Enclosing Box
    ####################################################################

    def enclosing_box(
            self,
            other: "BoundingBox2D",
    ) -> "BoundingBox2D":

        return BoundingBox2D(

            min(self.x1, other.x1),

            min(self.y1, other.y1),

            max(self.x2, other.x2),

            max(self.y2, other.y2),

        )

    ####################################################################
    # Generalized IoU (GIoU)
    ####################################################################

    def giou(
            self,
            other: "BoundingBox2D",
    ) -> float:

        iou = self.iou(other)

        enclosure = self.enclosing_box(other)

        enclosure_area = enclosure.area

        if enclosure_area == 0:
            return iou

        union = self.union(other)

        return iou - (
                (enclosure_area - union)
                /
                enclosure_area
        )

    ####################################################################
    # Center Distance
    ####################################################################

    def center_distance_squared(
            self,
            other: "BoundingBox2D",
    ) -> float:

        c1 = self.center
        c2 = other.center

        dx = c1.x - c2.x
        dy = c1.y - c2.y

        return dx * dx + dy * dy

    ####################################################################
    # Distance IoU (DIoU)
    ####################################################################

    def diou(
            self,
            other: "BoundingBox2D",
    ) -> float:

        iou = self.iou(other)

        enclosure = self.enclosing_box(other)

        c2 = (
                enclosure.width ** 2
                +
                enclosure.height ** 2
        )

        if c2 == 0:
            return iou

        rho2 = self.center_distance_squared(other)

        return iou - rho2 / c2

    ####################################################################
    # Complete IoU (CIoU)
    ####################################################################

    def ciou(
            self,
            other: "BoundingBox2D",
    ) -> float:

        diou = self.diou(other)

        if self.height == 0 or other.height == 0:
            return diou

        v = (
                4
                /
                (pi ** 2)
                *
                (
                        atan(self.width / self.height)
                        -
                        atan(other.width / other.height)
                ) ** 2
        )

        alpha = v / (1 - self.iou(other) + v + 1e-7)

        return diou - alpha * v

    ####################################################################
    # Transformations
    ####################################################################

    def translate(
        self,
        dx: float,
        dy: float,
    ):

        return BoundingBox2D(
            self.x1 + dx,
            self.y1 + dy,
            self.x2 + dx,
            self.y2 + dy,
        )

    def scale(
        self,
        sx: float,
        sy: float,
    ):

        return BoundingBox2D(
            self.x1 * sx,
            self.y1 * sy,
            self.x2 * sx,
            self.y2 * sy,
        )

    def clip(
        self,
        width: float,
        height: float,
    ):

        return BoundingBox2D(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(width, self.x2),
            min(height, self.y2),
        )

    ########################################################################
    # Image Utilities
    ########################################################################

    def normalize(
            self,
            image_width: float,
            image_height: float,
    ):

        return BoundingBox2D(

            self.x1 / image_width,
            self.y1 / image_height,
            self.x2 / image_width,
            self.y2 / image_height,

        )

    def denormalize(
            self,
            image_width: float,
            image_height: float,
    ):

        return BoundingBox2D(

            self.x1 * image_width,
            self.y1 * image_height,
            self.x2 * image_width,
            self.y2 * image_height,

        )

    ########################################################################
    # Expansion Utilities
    ########################################################################

    def expand(
            self,
            pixels: float,
    ):

        return BoundingBox2D(

            self.x1 - pixels,
            self.y1 - pixels,
            self.x2 + pixels,
            self.y2 + pixels,

        )

    def expand_percent(
            self,
            percent: float,
    ):

        dx = self.width * percent / 2

        dy = self.height * percent / 2

        return BoundingBox2D(

            self.x1 - dx,
            self.y1 - dy,
            self.x2 + dx,
            self.y2 + dy,

        )

    ########################################################################
    # Tracking Helpers
    ########################################################################

    def center_distance(
            self,
            other: "BoundingBox2D",
    ):

        return self.center.distance_to(
            other.center
        )

    def overlaps(
            self,
            other: "BoundingBox2D",
    ):

        return self.intersection(other) > 0

    def merge(
            self,
            other: "BoundingBox2D",
    ):

        return BoundingBox2D(

            min(self.x1, other.x1),

            min(self.y1, other.y1),

            max(self.x2, other.x2),

            max(self.y2, other.y2),

        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_tuple(self):
        return self.to_xyxy()

    def as_dict(self):

        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }

    ####################################################################
    # Coordinate Conversion
    ####################################################################

    def to_yolo(self):

        return (

            self.center.x,

            self.center.y,

            self.width,

            self.height,

        )

    @classmethod
    def from_yolo(
            cls,
            cx,
            cy,
            width,
            height,
    ):

        return cls.from_center(
            cx,
            cy,
            width,
            height,
        )

@dataclass(frozen=True, slots=True)
class BoundingBox3D(Geometry):
    """
    Axis-aligned 3D bounding box.

    Coordinates:

        (xmin, ymin, zmin)
               |
               |
               |
        (xmax, ymax, zmax)
    """

    x_min: float
    y_min: float
    z_min: float

    x_max: float
    y_max: float
    z_max: float

    @classmethod
    def from_xyzxyz(
            cls,
            x_min,
            y_min,
            z_min,
            x_max,
            y_max,
            z_max,
    ):
        return cls(
            x_min,
            y_min,
            z_min,
            x_max,
            y_max,
            z_max,
        )

    @classmethod
    def from_center(
            cls,
            cx,
            cy,
            cz,
            width,
            height,
            depth,
    ):
        return cls(

            cx - width / 2,

            cy - height / 2,

            cz - depth / 2,

            cx + width / 2,

            cy + height / 2,

            cz + depth / 2,
        )

    @property
    def width(self):
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self):
        return max(0.0, self.y_max - self.y_min)

    @property
    def depth(self):
        return max(0.0, self.z_max - self.z_min)

    @property
    def volume(self):
        return self.width * self.height * self.depth

    @property
    def center(self):
        return Point3D(

            (self.x_min + self.x_max) / 2,

            (self.y_min + self.y_max) / 2,

            (self.z_min + self.z_max) / 2,

        )

    @property
    def is_valid(self):
        return (

                self.x_max >= self.x_min

                and

                self.y_max >= self.y_min

                and

                self.z_max >= self.z_min

        )

    def contains_point(
            self,
            point: Point3D,
    ):
        return (

                self.x_min <= point.x <= self.x_max

                and

                self.y_min <= point.y <= self.y_max

                and

                self.z_min <= point.z <= self.z_max

        )

    def intersection(
            self,
            other: "BoundingBox3D",
    ):
        dx = max(
            0.0,
            min(self.x_max, other.x_max)
            -
            max(self.x_min, other.x_min),
        )

        dy = max(
            0.0,
            min(self.y_max, other.y_max)
            -
            max(self.y_min, other.y_min),
        )

        dz = max(
            0.0,
            min(self.z_max, other.z_max)
            -
            max(self.z_min, other.z_min),
        )

        return dx * dy * dz

    def union(
            self,
            other: "BoundingBox3D",
    ):
        return (

                self.volume

                +

                other.volume

                -

                self.intersection(other)

        )

    def iou(
            self,
            other: "BoundingBox3D",
    ):
        union = self.union(other)

        if union == 0:
            return 0.0

        return self.intersection(other) / union

    def translate(
            self,
            dx,
            dy,
            dz,
    ):
        return BoundingBox3D(

            self.x_min + dx,
            self.y_min + dy,
            self.z_min + dz,

            self.x_max + dx,
            self.y_max + dy,
            self.z_max + dz,

        )

    def scale(
            self,
            sx,
            sy,
            sz,
    ):
        return BoundingBox3D(

            self.x_min * sx,
            self.y_min * sy,
            self.z_min * sz,

            self.x_max * sx,
            self.y_max * sy,
            self.z_max * sz,

        )

    def as_tuple(self):
        return (

            self.x_min,
            self.y_min,
            self.z_min,

            self.x_max,
            self.y_max,
            self.z_max,

        )

    def as_dict(self):
        return {

            "x_min": self.x_min,
            "y_min": self.y_min,
            "z_min": self.z_min,

            "x_max": self.x_max,
            "y_max": self.y_max,
            "z_max": self.z_max,

        }


