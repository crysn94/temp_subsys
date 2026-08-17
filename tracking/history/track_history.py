"""
========================================================================
Track History
========================================================================

Maintains the measurement history for a tracked object.

The history is bounded to avoid unlimited memory growth.

========================================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from core.detection_result import DetectionResult


@dataclass(slots=True)
class TrackHistory:

    ####################################################################
    # Configuration
    ####################################################################

    max_length: int = 100

    ####################################################################
    # Storage
    ####################################################################

    detections: deque[DetectionResult] = field(init=False)

    ####################################################################

    def __post_init__(self):

        self.detections = deque(maxlen=self.max_length)

    ####################################################################
    # Collection Operations
    ####################################################################

    def append(
        self,
        detection: DetectionResult,
    ) -> None:

        self.detections.append(detection)

    ####################################################################

    def clear(self) -> None:

        self.detections.clear()

    ####################################################################

    def latest(self) -> DetectionResult | None:

        if not self.detections:
            return None

        return self.detections[-1]

    ####################################################################

    def oldest(self) -> DetectionResult | None:

        if not self.detections:
            return None

        return self.detections[0]

    ####################################################################
    # Motion Information
    ####################################################################

    def positions(self):

        return [

            d.position

            for d in self.detections

            if d.position is not None

        ]

    ####################################################################

    def velocities(self):

        return [

            d.velocity

            for d in self.detections

            if d.velocity is not None

        ]

    ####################################################################

    def timestamps(self):

        return [

            d.timestamp

            for d in self.detections

        ]

    ####################################################################

    def trajectory(self):

        """
        Alias for positions().
        """

        return self.positions()

    ####################################################################

    def duration(self):

        if len(self.detections) < 2:

            return None

        first = self.oldest().timestamp

        last = self.latest().timestamp

        return last.elapsed(first)

    ####################################################################
    # Statistics
    ####################################################################

    @property
    def size(self):

        return len(self.detections)

    ####################################################################

    @property
    def is_empty(self):

        return len(self.detections) == 0

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "count": len(self.detections),

            "duration":

                str(self.duration())

                if self.duration()

                else None,

            "detections": [

                d.as_dict()

                for d in self.detections

            ],

        }

    ####################################################################

    def __len__(self):

        return len(self.detections)

    ####################################################################

    def __iter__(self):

        return iter(self.detections)

    ####################################################################

    def __str__(self):

        return (

            f"TrackHistory("

            f"{len(self.detections)} detections)"

        )