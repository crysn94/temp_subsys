"""
========================================================================
Sensor Identifier
========================================================================

Canonical sensor description used throughout the C-UAS framework.

Every DetectionResult references one SensorIdentifier.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.geometry.pose import Pose3D


# ======================================================================
# Sensor Categories
# ======================================================================

class SensorCategory(str, Enum):

    EO = "eo"

    IR = "ir"

    RADAR = "radar"

    RF = "rf"

    ACOUSTIC = "acoustic"

    ADSB = "adsb"

    AIS = "ais"

    LIDAR = "lidar"

    FUSION = "fusion"

    UNKNOWN = "unknown"


# ======================================================================
# Capabilities
# ======================================================================

class SensorCapability(str, Enum):

    DETECTION = "detection"

    CLASSIFICATION = "classification"

    TRACKING = "tracking"

    RANGING = "ranging"

    VELOCITY = "velocity"

    IMAGING = "imaging"

    IDENTIFICATION = "identification"

    LOCALIZATION = "localization"


# ======================================================================
# Sensor Identifier
# ======================================================================

@dataclass(slots=True)
class SensorIdentifier:

    sensor_id: str

    name: str

    category: SensorCategory

    manufacturer: str = ""

    model: str = ""

    serial_number: str = ""

    firmware: str = ""

    pose: Pose3D | None = None

    enabled: bool = True

    capabilities: set[SensorCapability] = field(
        default_factory=set
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    ##################################################################

    @property
    def is_imaging(self) -> bool:

        return self.category in {

            SensorCategory.EO,

            SensorCategory.IR,

            SensorCategory.LIDAR,

        }

    ##################################################################

    @property
    def is_radar(self):

        return self.category == SensorCategory.RADAR

    ##################################################################

    @property
    def is_rf(self):

        return self.category == SensorCategory.RF

    ##################################################################

    def supports(

        self,

        capability: SensorCapability,

    ) -> bool:

        return capability in self.capabilities

    ##################################################################

    def add_capability(

        self,

        capability: SensorCapability,

    ):

        self.capabilities.add(capability)

    ##################################################################

    def remove_capability(

        self,

        capability: SensorCapability,

    ):

        self.capabilities.discard(capability)

    ##################################################################

    def as_dict(self):

        return {

            "sensor_id": self.sensor_id,

            "name": self.name,

            "category": self.category.value,

            "manufacturer": self.manufacturer,

            "model": self.model,

            "serial_number": self.serial_number,

            "firmware": self.firmware,

            "enabled": self.enabled,

            "capabilities": [

                c.value

                for c in self.capabilities

            ],

            "metadata": self.metadata,

        }