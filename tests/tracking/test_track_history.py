"""
Tests for Track History Management.

TrackHistory is responsible for maintaining historical snapshots of
a Track over time.

Responsibilities tested here:

- construction
- empty history behavior
- adding track snapshots
- retrieving history
- latest snapshot
- history length
- bounded history
- clearing history
- per-track history isolation
- deterministic ordering
- repeatability
- invalid input handling

TrackHistory does NOT own:

- Kalman filtering
- data association
- lifecycle transitions
- track management
- threat assessment
- sensor fusion

The Track object remains the canonical owner of the current track
state. TrackHistory stores historical state information.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.timestamps import Timestamp

from tracking.management.track_history import TrackHistory
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
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> Track:

    return Track(
        track_id=track_id,
        state_vector=make_state_vector(
            x=x,
            y=y,
            z=z,
        ),
    )


# ======================================================================
# Construction
# ======================================================================


def test_track_history_can_be_constructed():

    history = TrackHistory()

    assert history is not None


def test_default_history_is_empty():

    history = TrackHistory()

    assert history.num_tracks == 0
    assert history.num_entries == 0


# ======================================================================
# Add / Record
# ======================================================================


def test_record_track():

    history = TrackHistory()

    track = make_track(
        "T001",
        x=10.0,
    )

    result = history.record(track)

    assert result is not None
    assert history.num_tracks == 1
    assert history.num_entries == 1


def test_record_multiple_states_for_same_track():

    history = TrackHistory()

    track1 = make_track(
        "T001",
        x=10.0,
    )

    history.record(track1)

    track2 = make_track(
        "T001",
        x=20.0,
    )

    history.record(track2)

    assert history.num_tracks == 1
    assert history.num_entries == 2


# ======================================================================
# Retrieval
# ======================================================================


def test_get_history_for_track():

    history = TrackHistory()

    track1 = make_track(
        "T001",
        x=10.0,
    )

    track2 = make_track(
        "T001",
        x=20.0,
    )

    history.record(track1)
    history.record(track2)

    entries = history.get_history("T001")

    assert len(entries) == 2


def test_get_missing_track_history_returns_empty():

    history = TrackHistory()

    assert history.get_history("UNKNOWN") == []


def test_latest_returns_latest_entry():

    history = TrackHistory()

    first = make_track(
        "T001",
        x=10.0,
    )

    second = make_track(
        "T001",
        x=20.0,
    )

    history.record(first)
    history.record(second)

    latest = history.latest("T001")

    assert latest is not None
    assert latest.state_vector.position.x == pytest.approx(20.0)


def test_latest_missing_track_returns_none():

    history = TrackHistory()

    assert history.latest("UNKNOWN") is None


# ======================================================================
# Track Isolation
# ======================================================================


def test_histories_are_isolated_per_track():

    history = TrackHistory()

    track1 = make_track(
        "T001",
        x=10.0,
    )

    track2 = make_track(
        "T002",
        x=20.0,
    )

    history.record(track1)
    history.record(track2)

    assert len(history.get_history("T001")) == 1
    assert len(history.get_history("T002")) == 1

    assert (
        history.latest("T001").state_vector.position.x
        == pytest.approx(10.0)
    )

    assert (
        history.latest("T002").state_vector.position.x
        == pytest.approx(20.0)
    )


# ======================================================================
# Counts
# ======================================================================


def test_num_entries_counts_all_history_entries():

    history = TrackHistory()

    history.record(
        make_track("T001", x=1.0)
    )

    history.record(
        make_track("T001", x=2.0)
    )

    history.record(
        make_track("T002", x=3.0)
    )

    assert history.num_tracks == 2
    assert history.num_entries == 3


def test_num_entries_for_track():

    history = TrackHistory()

    history.record(
        make_track("T001", x=1.0)
    )

    history.record(
        make_track("T001", x=2.0)
    )

    history.record(
        make_track("T001", x=3.0)
    )

    assert history.num_entries_for("T001") == 3
    assert history.num_entries_for("UNKNOWN") == 0


# ======================================================================
# Ordering
# ======================================================================


def test_history_preserves_insertion_order():

    history = TrackHistory()

    history.record(
        make_track("T001", x=1.0)
    )

    history.record(
        make_track("T001", x=2.0)
    )

    history.record(
        make_track("T001", x=3.0)
    )

    entries = history.get_history("T001")

    positions = [
        entry.state_vector.position.x
        for entry in entries
    ]

    assert positions == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(3.0),
    ]


# ======================================================================
# Bounded History
# ======================================================================


def test_max_history_size_can_be_configured():

    history = TrackHistory(
        max_history_size=3,
    )

    assert history.max_history_size == 3


def test_history_is_bounded():

    history = TrackHistory(
        max_history_size=3,
    )

    for x in range(5):

        history.record(
            make_track(
                "T001",
                x=float(x),
            )
        )

    entries = history.get_history("T001")

    assert len(entries) == 3

    positions = [
        entry.state_vector.position.x
        for entry in entries
    ]

    assert positions == [
        pytest.approx(2.0),
        pytest.approx(3.0),
        pytest.approx(4.0),
    ]


def test_invalid_max_history_size_is_rejected():

    with pytest.raises(ValueError):

        TrackHistory(
            max_history_size=0,
        )


# ======================================================================
# Clear
# ======================================================================


def test_clear_track_history():

    history = TrackHistory()

    history.record(
        make_track("T001")
    )

    history.record(
        make_track("T002")
    )

    history.clear()

    assert history.num_tracks == 0
    assert history.num_entries == 0
    assert history.get_history("T001") == []
    assert history.get_history("T002") == []


def test_clear_single_track_history():

    history = TrackHistory()

    history.record(
        make_track("T001")
    )

    history.record(
        make_track("T002")
    )

    history.clear_track("T001")

    assert history.get_history("T001") == []
    assert len(history.get_history("T002")) == 1
    assert history.num_tracks == 1


# ======================================================================
# Input Validation
# ======================================================================


def test_record_requires_track():

    history = TrackHistory()

    with pytest.raises(TypeError):

        history.record(None)


def test_record_rejects_invalid_object():

    history = TrackHistory()

    with pytest.raises(TypeError):

        history.record("not-a-track")


# ======================================================================
# Returned Collections
# ======================================================================


def test_get_history_returns_copy():

    history = TrackHistory()

    history.record(
        make_track("T001")
    )

    entries = history.get_history("T001")

    entries.clear()

    assert len(history.get_history("T001")) == 1


# ======================================================================
# Contains
# ======================================================================


def test_contains_track():

    history = TrackHistory()

    assert not history.contains("T001")

    history.record(
        make_track("T001")
    )

    assert history.contains("T001")
    assert not history.contains("T002")


# ======================================================================
# Repeatability
# ======================================================================


def test_history_is_repeatable():

    def run_sequence():

        history = TrackHistory(
            max_history_size=3,
        )

        history.record(
            make_track("T001", x=1.0)
        )

        history.record(
            make_track("T001", x=2.0)
        )

        history.record(
            make_track("T001", x=3.0)
        )

        history.record(
            make_track("T001", x=4.0)
        )

        return [
            entry.state_vector.position.x
            for entry in history.get_history("T001")
        ]

    assert run_sequence() == run_sequence()


# ======================================================================
# Representation
# ======================================================================


def test_track_history_repr():

    history = TrackHistory()

    representation = repr(history)

    assert "TrackHistory" in representation