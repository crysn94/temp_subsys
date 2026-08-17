"""
Tests for the central Tracker.

Tracker coordinates:

    DetectionResult
          |
          v
      Association
          |
          v
      Track update
          |
          v
   TrackLifecycleManager
          |
          v
      TrackManager

Tracker does not own the Track collection or lifecycle rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.detection_result import DetectionResult
from core.geometry import Point3D
from core.sensor_identifier import (
    SensorIdentifier,
    SensorCategory,
)
from core.timestamps import Timestamp

from tracking.tracker import Tracker
from tracking.models.track import Track
from tracking.models.state_vector import StateVector


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


def make_detection(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    confidence: float = 0.95,
    class_id: int = 0,
    class_name: str = "drone",
    sensor_id: str = "TEST-SENSOR",
) -> DetectionResult:

    sensor = SensorIdentifier(
        sensor_id=sensor_id,
        name="Test Sensor",
        category=SensorCategory.EO,
    )

    return DetectionResult(
        sensor=sensor,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),
    )


class DummyAssociator:
    """
    Minimal deterministic associator for Tracker unit tests.
    """

    def __init__(
        self,
        matches=None,
        unmatched_tracks=None,
        unmatched_detections=None,
    ):
        self.matches = (
            matches
            if matches is not None
            else []
        )

        self.unmatched_tracks = (
            unmatched_tracks
            if unmatched_tracks is not None
            else []
        )

        self.unmatched_detections = (
            unmatched_detections
            if unmatched_detections is not None
            else []
        )

    def associate(
        self,
        tracks,
        detections,
    ):
        from tracking.association.base_associator import (
            AssociationResult,
        )

        return AssociationResult(
            matches=list(self.matches),
            unmatched_tracks=list(
                self.unmatched_tracks
            ),
            unmatched_detections=list(
                self.unmatched_detections
            ),
        )


# ======================================================================
# Construction
# ======================================================================


def test_tracker_can_be_constructed():

    tracker = Tracker()

    assert tracker is not None


def test_tracker_can_be_constructed_with_associator():

    associator = DummyAssociator()

    tracker = Tracker(
        associator=associator,
    )

    assert tracker.associator is associator


# ======================================================================
# Empty input
# ======================================================================


def test_empty_detections_do_not_crash():

    tracker = Tracker()

    result = tracker.update([])

    assert result is not None


def test_empty_tracker_has_no_tracks():

    tracker = Tracker()

    assert tracker.num_tracks == 0


# ======================================================================
# Track creation
# ======================================================================


def test_unmatched_detection_creates_track():

    tracker = Tracker()

    detection = make_detection(
        x=10.0,
        y=20.0,
        z=30.0,
    )

    result = tracker.update(
        [detection]
    )

    assert result is not None
    assert tracker.num_tracks == 1


def test_multiple_unmatched_detections_create_tracks():

    tracker = Tracker()

    detections = [
        make_detection(0.0, 0.0, 0.0),
        make_detection(10.0, 0.0, 0.0),
        make_detection(20.0, 0.0, 0.0),
    ]

    tracker.update(detections)

    assert tracker.num_tracks == 3


# ======================================================================
# Existing tracks
# ======================================================================


def test_existing_track_can_be_retrieved():

    tracker = Tracker()

    track = make_track("T001")

    tracker.add_track(track)

    assert tracker.get_track("T001") is track


def test_tracker_preserves_track_identity():

    tracker = Tracker()

    track = make_track("T001")

    tracker.add_track(track)

    assert tracker.get_track("T001") is track


# ======================================================================
# Association
# ======================================================================


def test_association_is_called():

    associator = DummyAssociator(
        matches=[(0, 0)]
    )

    tracker = Tracker(
        associator=associator,
    )

    tracker.add_track(
        make_track("T001")
    )

    detection = make_detection(
        1.0,
        0.0,
        0.0,
    )

    result = tracker.update(
        [detection]
    )

    assert result is not None


def test_one_to_one_association_is_preserved():

    associator = DummyAssociator(
        matches=[(0, 0)]
    )

    tracker = Tracker(
        associator=associator,
    )

    tracker.add_track(
        make_track("T001")
    )

    tracker.update(
        [make_detection()]
    )

    assert tracker.num_tracks == 1


# ======================================================================
# Track access
# ======================================================================


def test_tracks_property():

    tracker = Tracker()

    track = make_track("T001")

    tracker.add_track(track)

    assert tracker.tracks == [track]


def test_active_tracks_property():

    tracker = Tracker()

    track = make_track("T001")

    tracker.add_track(track)

    assert track in tracker.active_tracks


def test_num_tracks():

    tracker = Tracker()

    tracker.add_track(
        make_track("T001")
    )

    tracker.add_track(
        make_track("T002")
    )

    assert tracker.num_tracks == 2


# ======================================================================
# Remove
# ======================================================================


def test_remove_track():

    tracker = Tracker()

    track = make_track("T001")

    tracker.add_track(track)

    removed = tracker.remove_track(
        "T001"
    )

    assert removed is track
    assert tracker.num_tracks == 0


# ======================================================================
# Clear
# ======================================================================


def test_clear():

    tracker = Tracker()

    tracker.add_track(
        make_track("T001")
    )

    tracker.add_track(
        make_track("T002")
    )

    tracker.clear()

    assert tracker.num_tracks == 0


# ======================================================================
# Validation
# ======================================================================


def test_invalid_detections_are_rejected():

    tracker = Tracker()

    with pytest.raises(TypeError):

        tracker.update(
            [None]
        )


def test_invalid_track_is_rejected():

    tracker = Tracker()

    with pytest.raises(TypeError):

        tracker.add_track(
            None
        )


# ======================================================================
# Deterministic behavior
# ======================================================================


def test_tracker_is_repeatable():

    def run():

        tracker = Tracker()

        tracker.update(
            [
                make_detection(
                    0.0,
                    0.0,
                    0.0,
                ),
                make_detection(
                    10.0,
                    0.0,
                    0.0,
                ),
            ]
        )

        return tracker.num_tracks

    assert run() == run()