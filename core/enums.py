"""
========================================================================
C-UAS Core Enumerations
========================================================================

Common enumerations shared across the entire C-UAS Vision Subsystem.

These enums are intentionally detector-independent and can be reused by

• Vision
• Tracking
• Radar
• RF
• Acoustic
• Sensor Fusion
• Threat Assessment
• Mission Planning
• Mitigation

========================================================================
"""

from __future__ import annotations

from enum import Enum
from enum import IntEnum
from enum import auto


# ======================================================================
# Object Classification
# ======================================================================

class ObjectClass(IntEnum):
    """
    Canonical object classes.
    """

    UNKNOWN = 0

    BIRD = auto()

    DRONE = auto()

    HELICOPTER = auto()

    AIRCRAFT = auto()

    GLIDER = auto()

    PARAGLIDER = auto()

    HOT_AIR_BALLOON = auto()

    UAV_FIXED_WING = auto()

    UAV_MULTI_ROTOR = auto()

    UAV_HYBRID = auto()

    PERSON = auto()

    VEHICLE = auto()

    BOAT = auto()

    BUILDING = auto()

    OBSTACLE = auto()


# ======================================================================
# Detection Source
# ======================================================================

class DetectionSource(Enum):

    EO = "EO"

    IR = "IR"

    RADAR = "RADAR"

    RF = "RF"

    ACOUSTIC = "ACOUSTIC"

    ADSB = "ADSB"

    AIS = "AIS"

    FUSED = "FUSED"

    UNKNOWN = "UNKNOWN"


# ======================================================================
# Sensor Category
# ======================================================================

class SensorCategory(Enum):

    PASSIVE = auto()

    ACTIVE = auto()

    HYBRID = auto()


# ======================================================================
# Sensor Capability
# ======================================================================

class SensorCapability(Enum):

    DETECTION = auto()

    CLASSIFICATION = auto()

    TRACKING = auto()

    IDENTIFICATION = auto()

    RANGE_ESTIMATION = auto()

    VELOCITY_ESTIMATION = auto()

    GEOLOCATION = auto()


# ======================================================================
# Tracking
# ======================================================================

class TrackState(Enum):

    NEW = auto()

    TENTATIVE = auto()

    CONFIRMED = auto()

    LOST = auto()

    REIDENTIFIED = auto()

    TERMINATED = auto()


# ======================================================================
# Threat
# ======================================================================

class ThreatLevel(IntEnum):

    NONE = 0

    LOW = 1

    MEDIUM = 2

    HIGH = 3

    CRITICAL = 4


class ThreatCategory(Enum):

    UNKNOWN = auto()

    RECONNAISSANCE = auto()

    SURVEILLANCE = auto()

    LOITERING = auto()

    KINETIC = auto()

    SWARM = auto()


# ======================================================================
# Identity
# ======================================================================

class TargetIdentity(Enum):

    UNKNOWN = auto()

    FRIEND = auto()

    HOSTILE = auto()

    NEUTRAL = auto()

    CIVILIAN = auto()


# ======================================================================
# Mission Priority
# ======================================================================

class Priority(IntEnum):

    LOW = 1

    NORMAL = 2

    HIGH = 3

    URGENT = 4

    CRITICAL = 5


# ======================================================================
# Detection Status
# ======================================================================

class DetectionStatus(Enum):

    DETECTED = auto()

    TRACKED = auto()

    CLASSIFIED = auto()

    VERIFIED = auto()

    REJECTED = auto()


# ======================================================================
# Platform
# ======================================================================

class PlatformType(Enum):

    GROUND = auto()

    AIRBORNE = auto()

    NAVAL = auto()

    FIXED_INSTALLATION = auto()

    MOBILE = auto()


# ======================================================================
# Operating Environment
# ======================================================================

class EnvironmentType(Enum):

    DAY = auto()

    NIGHT = auto()

    THERMAL = auto()

    FOG = auto()

    RAIN = auto()

    SNOW = auto()

    MARITIME = auto()

    DESERT = auto()

    URBAN = auto()

    RURAL = auto()