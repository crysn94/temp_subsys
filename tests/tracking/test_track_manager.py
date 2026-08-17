"""
Tests for TrackManager.

TrackManager is responsible for managing the lifecycle of Track
objects after data association.

Responsibilities tested here:

- construction
- empty manager behavior
- adding tracks
- retrieving tracks
- removing tracks
- clearing tracks
- track counts
- active-track filtering
- unique track identifiers
- track identity preservation
- deterministic behavior
"""

from __future__ import annotations

import numpy as np
import pytest

from core.timestamps import Timestamp

from tracking.management.track_manager import TrackManager
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
    track_id: str | None = None,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> Track:

    state = make_state_vector(
        x=x,
        y=y,
        z=z,
    )

    if track_id is None:

        return Track(
            state_vector=state,
        )

    return Track(
        track_id=track_id,
        state_vector=state,
    )


# ======================================================================
# Construction
# ======================================================================


def test_track_manager_can_be_constructed():

    manager = TrackManager()

    assert manager is not None


# ======================================================================
# Empty manager
# ======================================================================


def test_empty_manager_contains_no_tracks():

    manager = TrackManager()

    assert manager.num_tracks == 0
    assert manager.tracks == []


def test_empty_manager_has_no_active_tracks():

    manager = TrackManager()

    assert manager.active_tracks == []


# ======================================================================
# Add
# ======================================================================


def test_add_track():

    manager = TrackManager()

    track = make_track("T001")

    result = manager.add_track(track)

    assert result is track
    assert manager.num_tracks == 1
    assert manager.get_track("T001") is track


def test_add_multiple_tracks():

    manager = TrackManager()

    track1 = make_track("T001")
    track2 = make_track("T002")
    track3 = make_track("T003")

    manager.add_track(track1)
    manager.add_track(track2)
    manager.add_track(track3)

    assert manager.num_tracks == 3

    assert manager.get_track("T001") is track1
    assert manager.get_track("T002") is track2
    assert manager.get_track("T003") is track3


# ======================================================================
# Duplicate IDs
# ======================================================================


def test_duplicate_track_id_is_rejected():

    manager = TrackManager()

    manager.add_track(
        make_track("T001")
    )

    with pytest.raises(ValueError):

        manager.add_track(
            make_track("T001")
        )


# ======================================================================
# Retrieval
# ======================================================================


def test_get_missing_track_returns_none():

    manager = TrackManager()

    assert manager.get_track("UNKNOWN") is None


def test_get_all_tracks_returns_tracks():

    manager = TrackManager()

    track1 = make_track("T001")
    track2 = make_track("T002")

    manager.add_track(track1)
    manager.add_track(track2)

    tracks = manager.get_tracks()

    assert len(tracks) == 2

    assert track1 in tracks
    assert track2 in tracks


# ======================================================================
# Remove
# ======================================================================


def test_remove_track():

    manager = TrackManager()

    track = make_track("T001")

    manager.add_track(track)

    removed = manager.remove_track("T001")

    assert removed is track
    assert manager.num_tracks == 0
    assert manager.get_track("T001") is None


def test_remove_missing_track_returns_none():

    manager = TrackManager()

    assert manager.remove_track("UNKNOWN") is None


# ======================================================================
# Clear
# ======================================================================


def test_clear_removes_all_tracks():

    manager = TrackManager()

    manager.add_track(
        make_track("T001")
    )

    manager.add_track(
        make_track("T002")
    )

    manager.add_track(
        make_track("T003")
    )

    manager.clear()

    assert manager.num_tracks == 0
    assert manager.tracks == []
    assert manager.active_tracks == []


# ======================================================================
# Track Identity
# ======================================================================


def test_track_manager_preserves_track_identity():

    manager = TrackManager()

    track = make_track("TRACK-123")

    manager.add_track(track)

    retrieved = manager.get_track("TRACK-123")

    assert retrieved is track
    assert retrieved.track_id == "TRACK-123"


# ======================================================================
# Active Tracks
# ======================================================================


def test_active_tracks_returns_active_tracks_only():

    manager = TrackManager()

    track1 = make_track("T001")
    track2 = make_track("T002")

    manager.add_track(track1)
    manager.add_track(track2)

    # --------------------------------------------------------------
    # Track lifecycle is controlled through the Track API.
    #
    # Track does not expose a writable "active" attribute.
    # --------------------------------------------------------------

    track1.confirm()
    track2.delete()

    active = manager.active_tracks

    assert track1 in active
    assert track2 not in active

    assert track1.is_active
    assert not track2.is_active


# ======================================================================
# Contains
# ======================================================================


def test_contains_track():

    manager = TrackManager()

    track = make_track("T001")

    manager.add_track(track)

    assert manager.contains("T001")
    assert not manager.contains("UNKNOWN")


# ======================================================================
# Deterministic ordering
# ======================================================================


def test_tracks_preserve_insertion_order():

    manager = TrackManager()

    track1 = make_track("T001")
    track2 = make_track("T002")
    track3 = make_track("T003")

    manager.add_track(track1)
    manager.add_track(track2)
    manager.add_track(track3)

    tracks = manager.get_tracks()

    assert tracks == [
        track1,
        track2,
        track3,
    ]


# ======================================================================
# Repeated Operations
# ======================================================================


def test_manager_is_repeatable():

    manager = TrackManager()

    track = make_track("T001")

    manager.add_track(track)

    assert manager.num_tracks == 1

    manager.remove_track("T001")

    assert manager.num_tracks == 0

    manager.add_track(track)

    assert manager.num_tracks == 1
    assert manager.get_track("T001") is track