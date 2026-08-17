"""
========================================================================
Track Lifecycle
========================================================================

Defines the lifecycle states of a tracked object.

Used by

• Track Manager
• Data Association
• Kalman Filter
• Sensor Fusion

========================================================================
"""

from __future__ import annotations

from enum import Enum, auto


class TrackState(Enum):
    """
    Lifecycle state of a tracked object.
    """

    NEW = auto()
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING = auto()
    LOST = auto()
    DELETED = auto()
    ARCHIVED = auto()

    ##################################################################
    # Properties
    ##################################################################

    @property
    def is_active(self) -> bool:
        """
        Track participates in tracking.
        """
        return self in {
            TrackState.NEW,
            TrackState.TENTATIVE,
            TrackState.CONFIRMED,
            TrackState.COASTING,
        }

    @property
    def is_confirmed(self) -> bool:
        return self == TrackState.CONFIRMED

    @property
    def is_terminal(self) -> bool:
        return self in {
            TrackState.DELETED,
            TrackState.ARCHIVED,
        }

    @property
    def can_receive_updates(self) -> bool:
        return self.is_active

    @property
    def is_lost(self) -> bool:
        return self in {
            TrackState.COASTING,
            TrackState.LOST,
        }

    def __str__(self) -> str:
        return self.name