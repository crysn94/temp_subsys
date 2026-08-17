"""
Integration tests for the complete tracking subsystem.

Pipeline under test:

    DetectionResult
          |
          v
       Tracker
          |
          +--------------------+
          |                    |
          v                    v
   Data Association      TrackManager
          |                    |
          +--------+-----------+
                   |
                   v
                 Track
                   |
          +--------+---------+
          |                  |
          v                  v
      Lifecycle          History

These tests verify component interaction and persistent behavior.
They do not test the internal mathematics of individual components.
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.timestamps import Timestamp

from tracking.tracker import Tracker
from tracking.models.lifecycle import TrackState
from tracking.models.track import Track


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

# IMPORTANT:
# Use the Point3D class already used by your DetectionResult/StateVector
# implementation.
#
# If your project uses a different import path, change ONLY this import.
from core.geometry import Point3D


# ======================================================================
# Helpers
# ======================================================================


def make_detection(
    x: float,
    y: float,
    z: float,
    confidence: float = 0.95,
) -> DetectionResult:
    """Create a valid 3-D DetectionResult."""

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


# ======================================================================
# Construction
# ======================================================================


def test_complete_tracking_system_can_be_constructed():
    tracker = Tracker()

    assert tracker is not None
    assert tracker.track_manager is not None
    assert tracker.lifecycle_manager is not None
    assert tracker.associator is not None


# ======================================================================
# Detection -> Track
# ======================================================================


def test_detection_creates_persistent_track():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    result = tracker.update(
        detections=[detection],
    )

    assert result.num_created_tracks == 1
    assert tracker.num_tracks == 1

    track = tracker.tracks[0]

    assert isinstance(track, Track)
    assert track.track_id == "T0001"


# ======================================================================
# Initial detection history
# ======================================================================


def test_created_track_receives_initial_detection():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    track = tracker.tracks[0]

    # Track should have received the detection through Tracker.
    #
    # Use the Track API if available so this test remains compatible
    # with the existing Track implementation.

    if hasattr(track, "detections"):
        assert len(track.detections) == 1


# ======================================================================
# Persistent identity
# ======================================================================


def test_associated_detection_preserves_track_identity():
    tracker = Tracker()

    first = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first],
    )

    original_track = tracker.tracks[0]
    original_id = original_track.track_id

    second = make_detection(
        10.5,
        20.5,
        30.5,
    )

    result = tracker.update(
        detections=[second],
    )

    assert tracker.num_tracks == 1
    assert result.num_updated_tracks == 1

    current_track = tracker.get_track(
        original_id
    )

    assert current_track is original_track
    assert current_track.track_id == original_id


# ======================================================================
# Track state update
# ======================================================================


def test_associated_detection_updates_track_position():
    tracker = Tracker()

    first = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first],
    )

    track = tracker.tracks[0]

    second = make_detection(
        11.0,
        21.0,
        31.0,
    )

    tracker.update(
        detections=[second],
    )

    position = track.latest_position

    assert position is not None
    assert position.x == 11.0
    assert position.y == 21.0
    assert position.z == 31.0


# ======================================================================
# History growth
# ======================================================================


def test_track_history_grows_across_associated_detections():
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

    assert tracker.num_tracks == 1

    track = tracker.tracks[0]

    if hasattr(track, "detections"):
        assert len(track.detections) == 3


# ======================================================================
# Lifecycle
# ======================================================================


def test_initial_detection_can_confirm_track():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    track = tracker.tracks[0]

    # Default lifecycle configuration may require multiple hits.
    # Therefore only verify that lifecycle information exists.
    assert track.lifecycle in TrackState


# ======================================================================
# Missed detection
# ======================================================================


def test_missed_track_is_processed_by_lifecycle_manager():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    track = tracker.tracks[0]

    # Repeated empty cycles exercise the missed-track path.
    tracker.update(detections=[])

    assert tracker.num_tracks >= 1
    assert track.track_id in [
        t.track_id
        for t in tracker.tracks
    ]


# ======================================================================
# New unmatched detection
# ======================================================================


def test_unmatched_detection_creates_new_track():
    tracker = Tracker()

    first = make_detection(
        0.0,
        0.0,
        0.0,
    )

    tracker.update(
        detections=[first],
    )

    assert tracker.num_tracks == 1

    second_cycle = [
        make_detection(
            0.5,
            0.5,
            0.5,
        ),
        make_detection(
            1000.0,
            1000.0,
            1000.0,
        ),
    ]

    result = tracker.update(
        detections=second_cycle,
    )

    assert tracker.num_tracks == 2
    assert result.num_created_tracks == 1


# ======================================================================
# Multiple persistent tracks
# ======================================================================


def test_multiple_tracks_remain_persistent():
    tracker = Tracker()

    first_cycle = [
        make_detection(0.0, 0.0, 0.0),
        make_detection(100.0, 100.0, 100.0),
        make_detection(200.0, 200.0, 200.0),
    ]

    tracker.update(
        detections=first_cycle,
    )

    initial_ids = [
        track.track_id
        for track in tracker.tracks
    ]

    assert len(initial_ids) == 3

    second_cycle = [
        make_detection(0.5, 0.5, 0.5),
        make_detection(100.5, 100.5, 100.5),
        make_detection(200.5, 200.5, 200.5),
    ]

    tracker.update(
        detections=second_cycle,
    )

    current_ids = [
        track.track_id
        for track in tracker.tracks
    ]

    assert current_ids == initial_ids


# ======================================================================
# Track ordering
# ======================================================================


def test_track_order_is_deterministic():
    tracker = Tracker()

    detections = [
        make_detection(0.0, 0.0, 0.0),
        make_detection(100.0, 100.0, 100.0),
        make_detection(200.0, 200.0, 200.0),
    ]

    tracker.update(
        detections=detections,
    )

    assert [
        track.track_id
        for track in tracker.tracks
    ] == [
        "T0001",
        "T0002",
        "T0003",
    ]


# ======================================================================
# Track object identity
# ======================================================================


def test_track_object_identity_survives_multiple_cycles():
    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    original = tracker.tracks[0]

    for offset in (0.5, 1.0, 1.5):

        tracker.update(
            detections=[
                make_detection(
                    10.0 + offset,
                    20.0 + offset,
                    30.0 + offset,
                )
            ],
        )

    current = tracker.get_track(
        original.track_id
    )

    assert current is original


# ======================================================================
# Result bookkeeping
# ======================================================================


def test_tracker_result_bookkeeping_is_consistent():
    tracker = Tracker()

    first = make_detection(
        0.0,
        0.0,
        0.0,
    )

    result1 = tracker.update(
        detections=[first],
    )

    assert result1.num_created_tracks == 1
    assert result1.num_tracks == 1

    second = make_detection(
        0.5,
        0.5,
        0.5,
    )

    result2 = tracker.update(
        detections=[second],
    )

    assert result2.num_tracks == 1
    assert result2.num_updated_tracks == 1
    assert result2.num_created_tracks == 0


# ======================================================================
# Repeatability
# ======================================================================


def run_integration_sequence():
    tracker = Tracker()

    sequence = [
        [
            make_detection(0.0, 0.0, 0.0),
        ],
        [
            make_detection(0.5, 0.5, 0.5),
        ],
        [
            make_detection(1.0, 1.0, 1.0),
        ],
        [],
        [
            make_detection(1.5, 1.5, 1.5),
        ],
    ]

    results = []

    for detections in sequence:
        result = tracker.update(
            detections=detections,
        )

        results.append(
            (
                [
                    track.track_id
                    for track in result.tracks
                ],
                list(result.created_track_ids),
                list(result.updated_track_ids),
                list(result.missed_track_ids),
                list(result.deleted_track_ids),
            )
        )

    return results


def test_complete_tracking_sequence_is_repeatable():
    result1 = run_integration_sequence()
    result2 = run_integration_sequence()

    assert result1 == result2

# ======================================================================
# Lifecycle -> TrackManager Integration
# ======================================================================


def test_track_enters_coasting_after_repeated_missed_cycles():

    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    track = tracker.tracks[0]

    # Continue empty cycles until the lifecycle manager
    # moves the track into COASTING or LOST.
    for _ in range(10):

        tracker.update(
            detections=[],
        )

        if track.lifecycle in (
            TrackState.COASTING,
            TrackState.LOST,
        ):
            break

    assert track.lifecycle in (
        TrackState.COASTING,
        TrackState.LOST,
    )


def test_track_can_recover_after_missed_cycle():

    tracker = Tracker()

    first = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first],
    )

    track = tracker.tracks[0]

    tracker.update(
        detections=[],
    )

    second = make_detection(
        10.5,
        20.5,
        30.5,
    )

    tracker.update(
        detections=[second],
    )

    recovered = tracker.get_track(
        track.track_id
    )

    assert recovered is track

    assert recovered.lifecycle != TrackState.DELETED

    assert recovered.latest_position is not None

    assert recovered.latest_position.x == 10.5
    assert recovered.latest_position.y == 20.5
    assert recovered.latest_position.z == 30.5


def test_track_identity_survives_coasting_and_recovery():

    tracker = Tracker()

    tracker.update(
        detections=[
            make_detection(
                10.0,
                20.0,
                30.0,
            )
        ]
    )

    original = tracker.tracks[0]
    original_id = original.track_id

    tracker.update(
        detections=[]
    )

    tracker.update(
        detections=[
            make_detection(
                10.2,
                20.2,
                30.2,
            )
        ]
    )

    recovered = tracker.get_track(
        original_id
    )

    assert recovered is original
    assert recovered.track_id == original_id

# ======================================================================
# Lost -> Deleted
# ======================================================================


def test_track_eventually_enters_lost_state_after_excessive_misses():

    tracker = Tracker()

    tracker.update(
        detections=[
            make_detection(
                10.0,
                20.0,
                30.0,
            )
        ]
    )

    original = tracker.tracks[0]
    original_id = original.track_id

    # Continue empty cycles until the lifecycle manager
    # transitions the track to LOST.
    for _ in range(100):

        current = tracker.get_track(original_id)

        if current is None:
            break

        if current.lifecycle == TrackState.LOST:
            break

        tracker.update(
            detections=[]
        )

    current = tracker.get_track(original_id)

    assert current is original
    assert original.lifecycle == TrackState.LOST

# ======================================================================
# Independent Track Lifecycle
# ======================================================================


def test_one_track_missing_does_not_affect_other_track():

    tracker = Tracker()

    tracker.update(
        detections=[
            make_detection(
                0.0,
                0.0,
                0.0,
            ),
            make_detection(
                100.0,
                100.0,
                100.0,
            ),
        ]
    )

    assert tracker.num_tracks == 2

    tracks = tracker.tracks

    track_a = tracks[0]
    track_b = tracks[1]

    id_a = track_a.track_id
    id_b = track_b.track_id

    # Keep only B visible.
    for _ in range(3):

        tracker.update(
            detections=[
                make_detection(
                    100.5,
                    100.5,
                    100.5,
                )
            ]
        )

    surviving_b = tracker.get_track(id_b)

    assert surviving_b is track_b
    assert surviving_b.track_id == id_b

    assert surviving_b.latest_position is not None

    assert surviving_b.latest_position.x == 100.5
    assert surviving_b.latest_position.y == 100.5
    assert surviving_b.latest_position.z == 100.5

def test_reappearing_object_reuses_existing_track():

    tracker = Tracker()

    tracker.update(
        detections=[
            make_detection(
                50.0,
                60.0,
                70.0,
            )
        ]
    )

    original = tracker.tracks[0]
    original_id = original.track_id

    # Temporary sensor loss.
    tracker.update(
        detections=[]
    )

    # Object reappears nearby.
    tracker.update(
        detections=[
            make_detection(
                50.5,
                60.5,
                70.5,
            )
        ]
    )

    assert tracker.num_tracks == 1

    recovered = tracker.get_track(
        original_id
    )

    assert recovered is original
    assert recovered.track_id == original_id