"""
========================================================================
Metadata Models
========================================================================

Provides structured metadata used throughout the C-UAS framework.

Metadata is intentionally generic so it can be attached to:

• DetectionResult
• Tracks
• Threats
• Missions
• Dataset Samples
• Replay Files

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, kw_only=True)
class Metadata:

    ####################################################################
    # Frame Information
    ####################################################################

    frame_id: str | None = None

    frame_number: int | None = None

    timestamp: datetime | None = None

    ####################################################################
    # Model
    ####################################################################

    model_name: str | None = None

    model_version: str | None = None

    ####################################################################
    # Processing
    ####################################################################

    latency_ms: float | None = None

    preprocessing_ms: float | None = None

    inference_ms: float | None = None

    postprocessing_ms: float | None = None

    ####################################################################
    # Mission / Recording
    ####################################################################

    mission: str | None = None

    recording: str | None = None

    operator: str | None = None

    ####################################################################
    # Environment
    ####################################################################

    weather: str | None = None

    illumination: str | None = None

    temperature_c: float | None = None

    ####################################################################
    # User-defined
    ####################################################################

    tags: set[str] = field(default_factory=set)

    custom: dict[str, Any] = field(default_factory=dict)

    ####################################################################
    # Helper Methods
    ####################################################################

    def add_tag(self, tag: str) -> None:
        self.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        self.tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def set(self, key: str, value: Any) -> None:
        self.custom[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.custom.get(key, default)

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self) -> dict[str, Any]:

        return {

            "frame_id": self.frame_id,

            "frame_number": self.frame_number,

            "timestamp":
                self.timestamp.isoformat()
                if self.timestamp else None,

            "model_name": self.model_name,

            "model_version": self.model_version,

            "latency_ms": self.latency_ms,

            "preprocessing_ms": self.preprocessing_ms,

            "inference_ms": self.inference_ms,

            "postprocessing_ms": self.postprocessing_ms,

            "mission": self.mission,

            "recording": self.recording,

            "operator": self.operator,

            "weather": self.weather,

            "illumination": self.illumination,

            "temperature_c": self.temperature_c,

            "tags": sorted(self.tags),

            "custom": self.custom,

        }

    ####################################################################

    def __str__(self):

        return (

            f"Metadata("
            f"frame={self.frame_number}, "
            f"model={self.model_name}, "
            f"latency={self.latency_ms} ms)"
        )