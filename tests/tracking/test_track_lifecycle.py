"""
Tests for Track Lifecycle Management.

The lifecycle manager is responsible for applying lifecycle rules
to Track objects.

Canonical lifecycle:

    NEW
      |
      v
  CONFIRMED
      |
      v
  COASTING
      |
      v
    LOST
      |
      v
  DELETED

The lifecycle manager coordinates transitions but does not own
the Track state itself.

The Track object remains the canonical owner of:

- lifecycle state
- hits
- misses
- consecutive hits
- consecutive misses
- timestamps
- state vector
"""

from __future__ import annotations

import numpy as np
import pytest

from core.timestamps import Timestamp

from tracking.management.track_lifecycle import TrackLifecycleManager
from tracking.models.lifecycle import TrackState
from tracking.models.state_vector import StateVector
from tracking.models.track import Track


# ======================================================================
# Helpers
# ======================================================================


def make_state_vector(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> StateVector:

    return StateVector.from_components(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        covariance=np.eye(6),
        timestamp=Timestamp.now(),
    )


def make_track(
    track_id: str = "T001",
) -> Track:

    return Track(
        track_id=track_id,
        state_vector=make_state_vector(),
    )


# ======================================================================
# Construction
# ======================================================================


def test_lifecycle_manager_can_be_constructed():

    manager = TrackLifecycleManager()

    assert manager is not None


def test_default_configuration_is_valid():

    manager = TrackLifecycleManager()

    assert manager.confirmation_hits > 0
    assert manager.coasting_misses > 0
    assert manager.lost_misses > 0


# ======================================================================
# Configuration Validation
# ======================================================================


def test_confirmation_hits_must_be_positive():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            confirmation_hits=0
        )


def test_coasting_misses_must_be_positive():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            coasting_misses=0
        )


def test_lost_misses_must_be_positive():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            lost_misses=0
        )


def test_lost_misses_must_not_be_less_than_coasting_misses():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            coasting_misses=3,
            lost_misses=2,
        )


# ======================================================================
# Initial State
# ======================================================================


def test_new_track_starts_in_new_state():

    track = make_track()

    assert track.lifecycle == TrackState.NEW


# ======================================================================
# Confirmation
# ======================================================================


def test_track_becomes_confirmed_after_required_hits():

    manager = TrackLifecycleManager(
        confirmation_hits=3,
    )

    track = make_track()

    manager.on_detection(track)
    assert track.lifecycle == TrackState.NEW

    manager.on_detection(track)
    assert track.lifecycle == TrackState.NEW

    manager.on_detection(track)
    assert track.lifecycle == TrackState.CONFIRMED


def test_confirmed_track_remains_confirmed_after_detection():

    manager = TrackLifecycleManager(
        confirmation_hits=2,
    )

    track = make_track()

    manager.on_detection(track)
    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED


# ======================================================================
# Missed Detections / Coasting
# ======================================================================


def test_confirmed_track_enters_coasting_after_misses():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
    )

    track = make_track()

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED

    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED

    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.COASTING


def test_coasting_track_returns_to_confirmed_after_detection():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
    )

    track = make_track()

    manager.on_detection(track)

    manager.on_missed_detection(track)
    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.COASTING

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED


# ======================================================================
# Lost
# ======================================================================


def test_coasting_track_becomes_lost_after_threshold():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
    )

    track = make_track()

    manager.on_detection(track)

    for _ in range(4):
        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST


def test_confirmed_track_can_become_lost_without_detection():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=3,
    )

    track = make_track()

    manager.on_detection(track)

    for _ in range(3):
        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST


# ======================================================================
# Recovery
# ======================================================================


def test_lost_track_can_recover_on_detection():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=1,
        lost_misses=3,
    )

    track = make_track()

    manager.on_detection(track)

    for _ in range(3):
        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED


# ======================================================================
# Deleted
# ======================================================================


def test_track_can_be_deleted():

    manager = TrackLifecycleManager()

    track = make_track()

    manager.delete(track)

    assert track.lifecycle == TrackState.DELETED


def test_deleted_track_remains_deleted():

    manager = TrackLifecycleManager()

    track = make_track()

    manager.delete(track)

    assert track.lifecycle == TrackState.DELETED

    manager.on_detection(track)

    assert track.lifecycle == TrackState.DELETED


def test_deleted_track_cannot_be_updated():

    manager = TrackLifecycleManager()

    track = make_track()

    manager.delete(track)

    with pytest.raises(ValueError):

        manager.on_missed_detection(track)


# ======================================================================
# Detection Reset
# ======================================================================


def test_detection_resets_consecutive_misses():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
    )

    track = make_track()

    manager.on_detection(track)

    manager.on_missed_detection(track)

    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.COASTING

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED


# ======================================================================
# Track Validation
# ======================================================================


def test_detection_requires_track():

    manager = TrackLifecycleManager()

    with pytest.raises(TypeError):

        manager.on_detection(None)


def test_missed_detection_requires_track():

    manager = TrackLifecycleManager()

    with pytest.raises(TypeError):

        manager.on_missed_detection(None)


def test_delete_requires_track():

    manager = TrackLifecycleManager()

    with pytest.raises(TypeError):

        manager.delete(None)


# ======================================================================
# Repeatability
# ======================================================================


def test_lifecycle_behavior_is_repeatable():

    def run_sequence():

        manager = TrackLifecycleManager(
            confirmation_hits=2,
            coasting_misses=2,
            lost_misses=4,
        )

        track = make_track()

        manager.on_detection(track)
        manager.on_detection(track)

        manager.on_missed_detection(track)
        manager.on_missed_detection(track)

        manager.on_detection(track)

        return track.lifecycle

    assert run_sequence() == run_sequence()

# ======================================================================
# Deletion Configuration
# ======================================================================


def test_deletion_misses_is_positive():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            deletion_misses=0,
        )


def test_deletion_misses_must_not_be_less_than_lost_misses():

    with pytest.raises(ValueError):

        TrackLifecycleManager(
            lost_misses=5,
            deletion_misses=4,
        )


def test_default_deletion_configuration_is_valid():

    manager = TrackLifecycleManager()

    assert manager.deletion_misses > 0
    assert manager.deletion_misses >= manager.lost_misses

# ======================================================================
# Automatic Deletion
# ======================================================================


def test_lost_track_becomes_deleted_after_deletion_threshold():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
        deletion_misses=6,
    )

    track = make_track()

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED

    for _ in range(4):

        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST

    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST

    manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.DELETED

def test_lost_track_can_recover_before_deletion_threshold():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=2,
        lost_misses=4,
        deletion_misses=6,
    )

    track = make_track()

    manager.on_detection(track)

    for _ in range(5):

        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.LOST

    manager.on_detection(track)

    assert track.lifecycle == TrackState.CONFIRMED

def test_automatically_deleted_track_cannot_recover():

    manager = TrackLifecycleManager(
        confirmation_hits=1,
        coasting_misses=1,
        lost_misses=2,
        deletion_misses=3,
    )

    track = make_track()

    manager.on_detection(track)

    for _ in range(3):

        manager.on_missed_detection(track)

    assert track.lifecycle == TrackState.DELETED

    manager.on_detection(track)

    assert track.lifecycle == TrackState.DELETED