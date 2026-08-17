"""
Tests for Hungarian track-to-detection association.
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.geometry.point import Point3D
from core.sensor_identifier import (
    SensorCategory,
    SensorIdentifier,
)

from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.association.hungarian import HungarianAssociator
from tracking.models.state_vector import StateVector
from tracking.models.track import Track
from tracking.similarity.euclidean import EuclideanMetric


# ======================================================================
# Fixtures / Helpers
# ======================================================================

def make_sensor() -> SensorIdentifier:
    return SensorIdentifier(
        sensor_id="TEST_SENSOR",
        name="Test Sensor",
        category=SensorCategory.EO,
    )


def make_detection(
    x: float,
    y: float,
    z: float = 0.0,
    confidence: float = 0.95,
) -> DetectionResult:

    return DetectionResult(
        sensor=make_sensor(),
        class_id=0,
        class_name="drone",
        confidence=confidence,
        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),
    )


def make_track(
    x: float,
    y: float,
    z: float = 0.0,
) -> Track:

    state = StateVector.from_components(
        x=x,
        y=y,
        z=z,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        covariance=np.eye(6),
    )

    return Track(
        state_vector=state,
    )


def make_associator(
    max_cost: float | None = None,
) -> HungarianAssociator:

    metric = EuclideanMetric()

    cost_builder = CostMatrixBuilder(
        metric=metric,
    )

    return HungarianAssociator(
        cost_builder=cost_builder,
        max_cost=max_cost,
    )


# ======================================================================
# Construction
# ======================================================================

def test_hungarian_can_be_constructed():

    associator = make_associator()

    assert associator is not None
    assert associator.metric_name == "EuclideanMetric"


# ======================================================================
# Empty Inputs
# ======================================================================

def test_empty_tracks():

    associator = make_associator()

    detections = [
        make_detection(10.0, 10.0),
        make_detection(20.0, 20.0),
    ]

    result = associator.associate(
        [],
        detections,
    )

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == [0, 1]

    assert result.cost_matrix.shape == (
        0,
        2,
    )


def test_empty_detections():

    associator = make_associator()

    tracks = [
        make_track(10.0, 10.0),
        make_track(20.0, 20.0),
    ]

    result = associator.associate(
        tracks,
        [],
    )

    assert result.matches == []
    assert result.unmatched_tracks == [0, 1]
    assert result.unmatched_detections == []

    assert result.cost_matrix.shape == (
        2,
        0,
    )


def test_empty_inputs():

    associator = make_associator()

    result = associator.associate(
        [],
        [],
    )

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []

    assert result.cost_matrix.shape == (
        0,
        0,
    )


# ======================================================================
# Single Association
# ======================================================================

def test_single_track_single_detection():

    associator = make_associator()

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        1.0,
        0.0,
        0.0,
    )

    result = associator.associate(
        [track],
        [detection],
    )

    assert result.matches == [
        (0, 0)
    ]

    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []

    assert result.cost_matrix.shape == (
        1,
        1,
    )

    assert np.isclose(
        result.cost_matrix[0, 0],
        1.0,
    )


# ======================================================================
# Multiple Tracks / Detections
# ======================================================================

def test_multiple_tracks_multiple_detections():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
        make_track(100.0, 100.0),
    ]

    detections = [
        make_detection(1.0, 0.0),
        make_detection(101.0, 100.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert len(result.matches) == 2

    assert (0, 0) in result.matches
    assert (1, 1) in result.matches

    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


# ======================================================================
# Unmatched Track
# ======================================================================

def test_unmatched_track():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
        make_track(100.0, 100.0),
    ]

    detections = [
        make_detection(1.0, 0.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert (0, 0) in result.matches

    assert result.unmatched_tracks == [1]
    assert result.unmatched_detections == []


# ======================================================================
# Unmatched Detection
# ======================================================================

def test_unmatched_detection():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
    ]

    detections = [
        make_detection(1.0, 0.0),
        make_detection(100.0, 100.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert (0, 0) in result.matches

    assert result.unmatched_tracks == []
    assert result.unmatched_detections == [1]


# ======================================================================
# Maximum Cost
# ======================================================================

def test_max_cost_rejects_far_assignment():

    associator = make_associator(
        max_cost=10.0,
    )

    track = make_track(
        0.0,
        0.0,
    )

    detection = make_detection(
        100.0,
        100.0,
    )

    result = associator.associate(
        [track],
        [detection],
    )

    assert result.matches == []

    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


# ======================================================================
# One-to-One Matching
# ======================================================================

def test_matching_is_one_to_one():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
        make_track(1.0, 0.0),
    ]

    detections = [
        make_detection(0.5, 0.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert len(result.matches) == 1

    matched_tracks = {
        track_index
        for track_index, _ in result.matches
    }

    matched_detections = {
        detection_index
        for _, detection_index in result.matches
    }

    assert len(matched_tracks) == 1
    assert len(matched_detections) == 1


# ======================================================================
# Result Properties
# ======================================================================

def test_result_properties():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
    ]

    detections = [
        make_detection(1.0, 0.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.num_matches == 1
    assert result.num_unmatched_tracks == 0
    assert result.num_unmatched_detections == 0
    assert result.has_matches is True


# ======================================================================
# Result Serialization
# ======================================================================

def test_result_as_dict():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
    ]

    detections = [
        make_detection(1.0, 0.0),
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    data = result.as_dict()

    assert isinstance(
        data,
        dict,
    )

    assert "matches" in data
    assert "unmatched_tracks" in data
    assert "unmatched_detections" in data

    assert len(data["matches"]) == 1

    match = data["matches"][0]

    assert match["track_index"] == 0
    assert match["detection_index"] == 0
    assert np.isclose(
        match["cost"],
        1.0,
    )


# ======================================================================
# Repeatability
# ======================================================================

def test_hungarian_is_repeatable():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0),
        make_track(100.0, 100.0),
    ]

    detections = [
        make_detection(2.0, 0.0),
        make_detection(102.0, 100.0),
    ]

    result_1 = associator.associate(
        tracks,
        detections,
    )

    result_2 = associator.associate(
        tracks,
        detections,
    )

    assert result_1.matches == result_2.matches

    assert result_1.unmatched_tracks == (
        result_2.unmatched_tracks
    )

    assert result_1.unmatched_detections == (
        result_2.unmatched_detections
    )

    assert np.array_equal(
        result_1.cost_matrix,
        result_2.cost_matrix,
    )