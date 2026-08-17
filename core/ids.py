"""
========================================================================
C-UAS Identifier Models
========================================================================

Canonical identifiers shared across the C-UAS framework.

Every subsystem should use these identifiers instead of generating UUIDs
directly.

Used by

• Vision
• Tracking
• Sensor Fusion
• Threat Assessment
• Mission Manager

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
from uuid import uuid4


# ======================================================================
# Base Identifier
# ======================================================================

@dataclass(frozen=True, slots=True)
class Identifier:
    """
    Base immutable identifier.
    """

    value: UUID

    @classmethod
    def create(cls):
        return cls(uuid4())

    def __str__(self):
        return str(self.value)


# ======================================================================
# Detection
# ======================================================================

@dataclass(frozen=True, slots=True)
class DetectionID(Identifier):
    pass


# ======================================================================
# Track
# ======================================================================

@dataclass(frozen=True, slots=True)
class TrackID(Identifier):
    pass


# ======================================================================
# Sensor
# ======================================================================

@dataclass(frozen=True, slots=True)
class SensorID(Identifier):
    pass


# ======================================================================
# Frame
# ======================================================================

@dataclass(frozen=True, slots=True)
class FrameID(Identifier):
    pass


# ======================================================================
# Threat
# ======================================================================

@dataclass(frozen=True, slots=True)
class ThreatID(Identifier):
    pass


# ======================================================================
# Mission
# ======================================================================

@dataclass(frozen=True, slots=True)
class MissionID(Identifier):
    pass