"""
End-to-end tracking pipeline tests.

These tests verify interaction between the major tracking components:

    DetectionResult
          |
          v
       Tracker
          |
          +--> Association
          |
          +--> TrackManager
          |
          +--> Track
          |
          +--> TrackHistory
          |
          +--> Lifecycle
          |
          v
    TrackerResult

The tests intentionally verify system behavior rather than
retesting individual component implementations.
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.timestamps import Timestamp

from tracking.tracker import Tracker


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
        position=np.array(
            [x, y, z],
            dtype=float,
        ),
        timestamp=Timestamp.now(),
    )


# ======================================================================
# Construction
# ======================================================================


def test_tracking_pipeline_can_be_constructed():

    tracker = Tracker()

    assert tracker is not None
    assert tracker.num_tracks == 0


# ======================================================================
# Initial detection
# ======================================================================


def test_initial_detection_enters_tracking_pipeline():

    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    result = tracker.update(
        detections=[detection],
    )

    assert result is not None
    assert result.num_tracks == 1
    assert result.num_created_tracks == 1

    assert tracker.num_tracks == 1


# ======================================================================
# Track identity
# ======================================================================


def test_track_identity_is_preserved():

    tracker = Tracker()

    first_detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first_detection],
    )

    original_track = tracker.tracks[0]
    original_id = original_track.track_id

    second_detection = make_detection(
        10.5,
        20.5,
        30.5,
    )

    tracker.update(
        detections=[second_detection],
    )

    current_track = tracker.get_track(
        original_id
    )

    assert current_track is original_track
    assert current_track.track_id == original_id


# ======================================================================
# Position update
# ======================================================================


def test_associated_detection_updates_track_position():

    tracker = Tracker()

    first_detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[first_detection],
    )

    second_detection = make_detection(
        11.0,
        21.0,
        31.0,
    )

    tracker.update(
        detections=[second_detection],
    )

    track = tracker.tracks[0]

    position = track.latest_position

    assert position is not None

    assert position.x == 11.0
    assert position.y == 21.0
    assert position.z == 31.0


# ======================================================================
# Multiple objects
# ======================================================================


def test_multiple_objects_are_tracked():

    tracker = Tracker()

    detections = [
        make_detection(
            10.0,
            20.0,
            30.0,
        ),
        make_detection(
            100.0,
            200.0,
            300.0,
        ),
        make_detection(
            500.0,
            600.0,
            700.0,
        ),
    ]

    result = tracker.update(
        detections=detections,
    )

    assert result.num_created_tracks == 3
    assert tracker.num_tracks == 3


# ======================================================================
# Stable IDs
# ======================================================================


def test_multiple_track_ids_remain_stable():

    tracker = Tracker()

    first_cycle = [
        make_detection(10.0, 20.0, 30.0),
        make_detection(100.0, 200.0, 300.0),
    ]

    tracker.update(
        detections=first_cycle,
    )

    initial_ids = [
        track.track_id
        for track in tracker.tracks
    ]

    second_cycle = [
        make_detection(11.0, 21.0, 31.0),
        make_detection(101.0, 201.0, 301.0),
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
# New object
# ======================================================================


def test_new_unmatched_object_creates_new_track():

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

    assert tracker.num_tracks == 1

    tracker.update(
        detections=[
            make_detection(
                10.5,
                20.5,
                30.5,
            ),
            make_detection(
                500.0,
                600.0,
                700.0,
            ),
        ]
    )

    assert tracker.num_tracks == 2


# ======================================================================
# Empty detection cycle
# ======================================================================


def test_empty_detection_cycle_is_handled():

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

    assert tracker.num_tracks == 1

    result = tracker.update(
        detections=[]
    )

    assert result is not None
    assert result.num_missed_tracks >= 1


# ======================================================================
# Reappearance
# ======================================================================


def test_track_can_receive_detection_after_missed_cycle():

    tracker = Tracker()

    detection = make_detection(
        10.0,
        20.0,
        30.0,
    )

    tracker.update(
        detections=[detection],
    )

    track_id = tracker.tracks[0].track_id

    tracker.update(
        detections=[],
    )

    recovered_detection = make_detection(
        10.5,
        20.5,
        30.5,
    )

    tracker.update(
        detections=[recovered_detection],
    )

    assert tracker.get_track(
        track_id
    ) is not None


# ======================================================================
# Detection history
# ======================================================================


def test_tracking_pipeline_preserves_detection_history():

    tracker = Tracker()

    detections = [
        make_detection(10.0, 20.0, 30.0),
        make_detection(11.0, 21.0, 31.0),
        make_detection(12.0, 22.0, 32.0),
    ]

    for detection in detections:

        tracker.update(
            detections=[detection],
        )

    track = tracker.tracks[0]

    history = getattr(
        track,
        "detection_history",
        None,
    )

    assert history is not None

    assert len(history) >= 1


# ======================================================================
# TrackerResult consistency
# ======================================================================


def test_tracker_result_matches_tracker_state():

    tracker = Tracker()

    result = tracker.update(
        detections=[
            make_detection(
                10.0,
                20.0,
                30.0,
            ),
            make_detection(
                100.0,
                200.0,
                300.0,
            ),
        ]
    )

    assert result.num_tracks == tracker.num_tracks

    assert (
        result.num_active_tracks
        == tracker.num_active_tracks
    )

    result_ids = [
        track.track_id
        for track in result.tracks
    ]

    tracker_ids = [
        track.track_id
        for track in tracker.tracks
    ]

    assert result_ids == tracker_ids


# ======================================================================
# Determinism
# ======================================================================


def run_pipeline():

    tracker = Tracker()

    sequence = [
        [
            make_detection(
                10.0,
                20.0,
                30.0,
            )
        ],
        [
            make_detection(
                11.0,
                21.0,
                31.0,
            )
        ],
        [
            make_detection(
                12.0,
                22.0,
                32.0,
            )
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
                list(
                    result.created_track_ids
                ),
                list(
                    result.updated_track_ids
                ),
                list(
                    result.missed_track_ids
                ),
            )
        )

    return results


def test_tracking_pipeline_is_repeatable():

    result1 = run_pipeline()
    result2 = run_pipeline()

    assert result1 == result2