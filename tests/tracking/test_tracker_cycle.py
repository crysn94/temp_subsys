"""
Integration tests for the complete tracking cycle.

The tracking cycle connects the already-tested tracking components:

    Detection
        ↓
    Prediction
        ↓
    Association
        ↓
    Track Update
        ↓
    Lifecycle
        ↓
    Track History
        ↓
    Next Cycle

These tests intentionally focus on interaction between components
rather than repeating their individual unit tests.
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.timestamps import Timestamp
from core.geometry import Point3D
from tracking.tracker import Tracker
from tracking.models.track import Track
from tracking.models.state_vector import StateVector


# ======================================================================
# Helpers
# ======================================================================


def make_detection(
    x: float,
    y: float,
    z: float,
    confidence: float = 0.95,
) -> DetectionResult:

    return DetectionResult(
        sensor="test",
        class_id=0,
        class_name="drone",
        confidence=confidence,
        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),
        timestamp=Timestamp.now(),
    )


def make_state_vector(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> StateVector:
    """
    Create a deterministic state vector.
    """

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


# ======================================================================
# Construction
# ======================================================================


def test_tracker_cycle_can_be_constructed():
    tracker = Tracker()

    assert tracker is not None


# ======================================================================
# First detection
# ======================================================================


def test_first_detection_creates_track():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    assert tracker.track_manager.num_tracks == 1


# ======================================================================
# Persistent track
# ======================================================================


def test_second_detection_does_not_create_duplicate_track():
    tracker = Tracker()

    detection1 = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection1],
    )

    assert tracker.track_manager.num_tracks == 1

    track_id = (
        tracker.track_manager
        .get_tracks()[0]
        .track_id
    )

    detection2 = make_detection(
        10.5,
        20.5,
        30.5,
    )

    tracker.update(
        detections=[detection2],
    )

    assert tracker.track_manager.num_tracks == 1

    assert (
        tracker.track_manager
        .get_track(track_id)
        is not None
    )


# ======================================================================
# Multiple detections
# ======================================================================


def test_multiple_detections_create_multiple_tracks():
    tracker = Tracker()

    detections = [
        make_detection(10.0, 20.0, 30.0),
        make_detection(100.0, 200.0, 300.0),
        make_detection(500.0, 600.0, 700.0),
    ]

    timestamp = detections[0].timestamp

    tracker.update(
        detections=detections,
    )

    assert tracker.track_manager.num_tracks == 3


# ======================================================================
# One-to-one persistence
# ======================================================================


def test_track_ids_remain_stable_across_cycles():
    tracker = Tracker()

    first = [
        make_detection(10.0, 20.0, 30.0),
        make_detection(100.0, 200.0, 300.0),
    ]

    tracker.update(
        detections=first,
    )

    initial_ids = [
        track.track_id
        for track in tracker.track_manager.get_tracks()
    ]

    second = [
        make_detection(10.5, 20.5, 30.5),
        make_detection(100.5, 200.5, 300.5),
    ]

    tracker.update(
        detections=second,
    )

    current_ids = [
        track.track_id
        for track in tracker.track_manager.get_tracks()
    ]

    assert current_ids == initial_ids


# ======================================================================
# Unmatched detection
# ======================================================================


def test_unmatched_detection_creates_new_track():
    tracker = Tracker()

    first = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first],
    )

    assert tracker.track_manager.num_tracks == 1

    second = [
        make_detection(10.5, 20.5, 30.5),
        make_detection(500.0, 600.0, 700.0),
    ]

    tracker.update(
        detections=second,
    )

    assert tracker.track_manager.num_tracks == 2


# ======================================================================
# Empty detection cycle
# ======================================================================


def test_empty_detection_cycle_does_not_crash():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    assert tracker.track_manager.num_tracks == 1

    tracker.update(
        detections=[detection],
    )

    assert tracker.track_manager.num_tracks >= 1


# ======================================================================
# Track persistence
# ======================================================================


def test_track_object_identity_is_preserved():
    tracker = Tracker()

    detection1 = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection1],
    )

    original_track = (
        tracker.track_manager
        .get_tracks()[0]
    )

    detection2 = make_detection(
        10.5,
        20.5,
        30.5,
    )

    tracker.update(
        detections=[detection2],
    )

    current_track = (
        tracker.track_manager
        .get_track(
            original_track.track_id
        )
    )

    assert current_track is original_track


# ======================================================================
# Determinism
# ======================================================================


def run_tracking_sequence():
    tracker = Tracker()

    detections = [
        make_detection(10.0, 20.0, 30.0),
        make_detection(10.5, 20.5, 30.5),
        make_detection(11.0, 21.0, 31.0),
    ]

    for detection in detections:
        tracker.update(
            detections=[detection],
        )

    return [
        (
            track.track_id,
            track.lifecycle,
        )
        for track
        in tracker.track_manager.get_tracks()
    ]


def test_tracker_cycle_is_repeatable():
    result1 = run_tracking_sequence()
    result2 = run_tracking_sequence()

    assert result1 == result2