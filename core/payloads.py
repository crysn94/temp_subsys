"""
========================================================================
Sensor Payload Models
========================================================================

Each sensor produces a specialized payload.

DetectionResult stores ONE payload object.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.geometry.bbox import BoundingBox2D
from core.geometry.point import Point2D
from core.geometry.polygon import Polygon2D


# ======================================================================
# Base Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class Payload:

    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):

        return {
            "metadata": self.metadata
        }


# ======================================================================
# EO / IR Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class EOIRPayload(Payload):

    bbox: BoundingBox2D | None = None

    segmentation: Polygon2D | None = None

    centroid: Point2D | None = None

    image_width: int | None = None

    image_height: int | None = None

    confidence: float | None = None


# ======================================================================
# Radar Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class RadarPayload(Payload):

    range_m: float | None = None

    azimuth_deg: float | None = None

    elevation_deg: float | None = None

    radial_velocity: float | None = None

    snr: float | None = None

    rcs: float | None = None


# ======================================================================
# RF Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class RFPayload(Payload):

    frequency_hz: float | None = None

    bandwidth_hz: float |None = None

    doa_deg: float | None = None

    protocol: str | None = None

    modulation: str | None = None


# ======================================================================
# Acoustic Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class AcousticPayload(Payload):

    doa_deg: float | None = None

    confidence: float | None = None

    peak_frequency: float | None = None


# ======================================================================
# ADS-B Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class ADSBPayload(Payload):

    icao: str | None = None

    callsign: str | None = None

    altitude: float | None = None

    groundspeed: float | None = None

    heading: float | None = None


# ======================================================================
# AIS Payload
# ======================================================================

@dataclass(slots=True, kw_only=True)
class AISPayload(Payload):

    mmsi: str | None = None

    vessel_name: str | None = None

    sog: float | None = None

    cog: float | None = None