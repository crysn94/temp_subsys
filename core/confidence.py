"""
========================================================================
Confidence Models
========================================================================

Standard confidence representation for all perception modules.

Used by

• Detection
• Classification
• Tracking
• Sensor Fusion
• Threat Assessment

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Confidence:

    ####################################################################
    # Core Confidence Scores
    ####################################################################

    detection: float = 1.0

    classification: float = 1.0

    localization: float = 1.0

    tracking: float = 1.0

    identification: float = 1.0

    fusion: float = 1.0

    ####################################################################
    # Validation
    ####################################################################

    def __post_init__(self):

        for name, value in self.as_dict().items():

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} confidence must be in [0,1]"
                )

    ####################################################################
    # Overall Score
    ####################################################################

    @property
    def overall(self) -> float:

        values = tuple(self.as_dict().values())

        return sum(values) / len(values)

    ####################################################################
    # Statistics
    ####################################################################

    @property
    def minimum(self):

        return min(self.as_dict().values())

    @property
    def maximum(self):

        return max(self.as_dict().values())

    ####################################################################
    # Reliability
    ####################################################################

    def is_reliable(

        self,

        threshold: float = 0.70,

    ) -> bool:

        return self.overall >= threshold

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "detection": self.detection,

            "classification": self.classification,

            "localization": self.localization,

            "tracking": self.tracking,

            "identification": self.identification,

            "fusion": self.fusion,

        }

    ####################################################################

    def __str__(self):

        return (

            f"Confidence("
            f"overall={self.overall:.3f}, "
            f"detection={self.detection:.2f}, "
            f"classification={self.classification:.2f})"

        )