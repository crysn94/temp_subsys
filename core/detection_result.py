"""
========================================================================
Canonical Detection Result
========================================================================

Every perception module in the C-UAS framework outputs DetectionResult.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from core.timestamps import (Timestamp)
from enum import Enum
from typing import Any
from uuid import uuid4

from core.geometry.point import Point3D
from core.geometry.pose import Pose3D
from core.geometry.velocity import Velocity3D
from core.geometry.acceleration import Acceleration3D
from core.geometry.bbox import BoundingBox2D

from core.payloads import Payload

from core.sensor_identifier import SensorIdentifier


# ======================================================================
# Detection Status
# ======================================================================

class DetectionStatus(str, Enum):

    NEW = "new"

    TRACKED = "tracked"

    LOST = "lost"

    FUSED = "fused"

    CONFIRMED = "confirmed"

    REJECTED = "rejected"


# ======================================================================
# Detection Result
# ======================================================================

@dataclass(slots=True, kw_only=True)
class DetectionResult:

    ####################################################################
    # Identity
    ####################################################################

    detection_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    track_id: str | None = None

    timestamp: Timestamp = field(
        default_factory=Timestamp.now
    )

    ####################################################################
    # Sensor
    ####################################################################

    sensor: SensorIdentifier

    ####################################################################
    # Classification
    ####################################################################

    class_id: int

    class_name: str

    confidence: float

    ####################################################################
    # Geometry
    ####################################################################

    position: Point3D | None = None

    pose: Pose3D | None = None

    bbox: BoundingBox2D | None = None

    ####################################################################
    # Motion
    ####################################################################

    velocity: Velocity3D | None = None

    acceleration: Acceleration3D | None = None

    ####################################################################
    # Payload
    ####################################################################

    payload: Payload | None = None

    ####################################################################
    # Lifecycle
    ####################################################################

    status: DetectionStatus = DetectionStatus.NEW

    is_confirmed: bool = False

    is_fused: bool = False

    ####################################################################
    # Metadata
    ####################################################################

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    ####################################################################
    # Validation
    ####################################################################

    def validate(self):

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

    ####################################################################
    # Convenience
    ####################################################################

    @property
    def has_position(self):

        return self.position is not None

    @property
    def has_bbox(self) -> bool:
        return self.bbox is not None

    @property
    def has_pose(self):

        return self.pose is not None

    @property
    def has_velocity(self):

        return self.velocity is not None

    @property
    def has_payload(self):

        return self.payload is not None

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "detection_id": self.detection_id,

            "track_id": self.track_id,

            "timestamp": self.timestamp.as_dict(),

            "sensor": self.sensor.as_dict(),

            "class_id": self.class_id,

            "class_name": self.class_name,

            "confidence": self.confidence,

            "position": (
                self.position.as_dict()
                if self.position
                else None
            ),

            "pose": (
                self.pose.as_dict()
                if self.pose
                else None
            ),

            "bbox": (
                self.bbox.as_dict()
                if self.bbox
                else None
            ),

            "velocity": (
                self.velocity.as_tuple()
                if self.velocity
                else None
            ),

            "acceleration": (
                self.acceleration.as_tuple()
                if self.acceleration
                else None
            ),

            "payload": (
                self.payload.as_dict()
                if self.payload
                else None
            ),

            "status": self.status.value,

            "is_confirmed": self.is_confirmed,

            "is_fused": self.is_fused,

            "metadata": self.metadata,

        }

    ####################################################################

    def __str__(self):

        return (

            f"DetectionResult("
            f"id={self.detection_id}, "
            f"class={self.class_name}, "
            f"confidence={self.confidence:.2f}, "
            f"sensor={self.sensor.name})"
        )