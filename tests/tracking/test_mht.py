"""
========================================================================
MHT Association Tests
========================================================================

Tests the Multiple Hypothesis Tracking association layer.

Coverage
--------
• Construction
• Empty inputs
• Single track / detection
• Multiple tracks / detections
• One-to-one matching
• Multiple hypotheses
• Maximum hypothesis limit
• Cost gating
• Miss handling
• Clutter handling
• Hypothesis ordering
• Score validity
• Repeatability
• Track identity preservation
• Track state preservation
• Detection without position

========================================================================
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.sensor_identifier import (
    SensorCategory,
    SensorIdentifier,
)
from core.timestamps import Timestamp

from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.association.mht import (
    MHTAssociator,
    MHTHypothesis,
    MHTResult,
)
from tracking.models.state_vector import StateVector
from tracking.models.track import Track
from tracking.similarity.euclidean import EuclideanMetric


# ======================================================================
# Test Fixtures / Helpers
# ======================================================================

def make_sensor() -> SensorIdentifier:
    return SensorIdentifier(
        sensor_id="TEST_SENSOR",
        name="Test Sensor",
        category=SensorCategory.RADAR,
    )


def make_detection(
    x: float,
    y: float,
    z: float = 0.0,
) -> DetectionResult:

    return DetectionResult(
        timestamp=Timestamp.now(),
        sensor=make_sensor(),
        class_id=0,
        class_name="drone",
        confidence=0.9,
        position=__import__(
            "core.geometry.point",
            fromlist=["Point3D"],
        ).Point3D(
            x=x,
            y=y,
            z=z,
        ),
    )


def make_track(
    x: float,
    y: float,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> Track:

    state = StateVector.from_components(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        covariance=np.eye(6),
        timestamp=Timestamp.now(),
    )

    track = Track(
        state_vector=state,
    )

    return track


def make_associator(
    *,
    max_hypotheses: int = 20,
    max_cost: float | None = None,
    miss_cost: float = 5.0,
    clutter_cost: float = 5.0,
) -> MHTAssociator:

    metric = EuclideanMetric()

    cost_builder = CostMatrixBuilder(
        metric=metric,
    )

    return MHTAssociator(
        cost_builder=cost_builder,
        max_hypotheses=max_hypotheses,
        max_cost=max_cost,
        miss_cost=miss_cost,
        clutter_cost=clutter_cost,
    )


# ======================================================================
# Construction
# ======================================================================

def test_mht_can_be_constructed():

    associator = make_associator()

    assert associator is not None
    assert isinstance(
        associator,
        MHTAssociator,
    )


# ======================================================================
# Empty Inputs
# ======================================================================

def test_mht_empty_tracks():

    associator = make_associator()

    result = associator.associate(
        tracks=[],
        detections=[
            make_detection(0.0, 0.0, 0.0),
        ],
    )

    assert isinstance(
        result,
        MHTResult,
    )

    assert result.cost_matrix.shape == (0, 1)

    assert isinstance(
        result.hypotheses,
        list,
    )


def test_mht_empty_detections():

    associator = make_associator()

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[],
    )

    assert isinstance(
        result,
        MHTResult,
    )

    assert result.cost_matrix.shape == (1, 0)

    assert isinstance(
        result.hypotheses,
        list,
    )


def test_mht_empty_inputs():

    associator = make_associator()

    result = associator.associate(
        tracks=[],
        detections=[],
    )

    assert isinstance(
        result,
        MHTResult,
    )

    assert result.cost_matrix.shape == (0, 0)


# ======================================================================
# Single Track / Detection
# ======================================================================

def test_single_track_single_detection():

    associator = make_associator()

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        0.5,
        0.0,
        0.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[detection],
    )

    assert isinstance(
        result,
        MHTResult,
    )

    assert len(result.hypotheses) > 0

    for hypothesis in result.hypotheses:

        assert isinstance(
            hypothesis,
            MHTHypothesis,
        )


# ======================================================================
# Matching
# ======================================================================

def test_best_hypothesis_prefers_nearest_detection():

    associator = make_associator()

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    near_detection = make_detection(
        0.5,
        0.0,
        0.0,
    )

    far_detection = make_detection(
        50.0,
        0.0,
        0.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[
            near_detection,
            far_detection,
        ],
    )

    assert len(result.hypotheses) > 0

    best = result.hypotheses[0]

    assert best.total_cost >= 0.0

    if best.matches:
        assert best.matches[0][1] == 0


# ======================================================================
# Multiple Tracks / Detections
# ======================================================================

def test_multiple_tracks_multiple_detections():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(10.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.2, 0.0, 0.0),
        make_detection(10.2, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    assert len(result.hypotheses) > 0

    best = result.hypotheses[0]

    assert len(best.matches) <= 2


# ======================================================================
# One-to-One Constraint
# ======================================================================

def test_mht_enforces_one_to_one_matching():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(0.2, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.1, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    for hypothesis in result.hypotheses:

        detection_indices = [
            detection_index
            for _, detection_index
            in hypothesis.matches
        ]

        assert len(
            detection_indices
        ) == len(
            set(detection_indices)
        )


# ======================================================================
# Multiple Hypotheses
# ======================================================================

def test_mht_generates_multiple_hypotheses():

    associator = make_associator(
        max_hypotheses=20,
    )

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(1.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.5, 0.0, 0.0),
        make_detection(0.6, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    assert len(
        result.hypotheses
    ) > 1


# ======================================================================
# Maximum Hypothesis Limit
# ======================================================================

def test_mht_respects_max_hypotheses():

    max_hypotheses = 3

    associator = make_associator(
        max_hypotheses=max_hypotheses,
    )

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(1.0, 0.0, 0.0),
        make_track(2.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.2, 0.0, 0.0),
        make_detection(1.2, 0.0, 0.0),
        make_detection(2.2, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    assert len(
        result.hypotheses
    ) <= max_hypotheses


# ======================================================================
# Hypothesis Ordering
# ======================================================================

def test_mht_hypotheses_are_sorted_by_cost():

    associator = make_associator(
        max_hypotheses=20,
    )

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(10.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.2, 0.0, 0.0),
        make_detection(10.2, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    costs = [
        hypothesis.total_cost
        for hypothesis in result.hypotheses
    ]

    assert costs == sorted(costs)


# ======================================================================
# Score Validity
# ======================================================================

def test_mht_hypothesis_scores_are_finite():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(1.0, 0.0, 0.0),
    ]

    result = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    for hypothesis in result.hypotheses:

        assert np.isfinite(
            hypothesis.total_cost
        )

        assert np.isfinite(
            hypothesis.score
        )


def test_mht_hypothesis_scores_are_non_negative():

    associator = make_associator()

    result = associator.associate(
        tracks=[
            make_track(0.0, 0.0, 0.0),
        ],
        detections=[
            make_detection(1.0, 0.0, 0.0),
        ],
    )

    for hypothesis in result.hypotheses:

        assert hypothesis.score >= 0.0
        assert hypothesis.total_cost >= 0.0


# ======================================================================
# Miss Handling
# ======================================================================

def test_mht_can_create_missed_track_hypothesis():

    associator = make_associator(
        miss_cost=2.0,
    )

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        100.0,
        0.0,
        0.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[detection],
    )

    assert len(result.hypotheses) > 0

    assert any(
        len(hypothesis.unmatched_tracks) > 0
        for hypothesis in result.hypotheses
    )


# ======================================================================
# Clutter Handling
# ======================================================================

def test_mht_can_create_clutter_hypothesis():

    associator = make_associator(
        clutter_cost=2.0,
    )

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        100.0,
        100.0,
        100.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[detection],
    )

    assert len(result.hypotheses) > 0

    assert any(
        len(hypothesis.unmatched_detections) > 0
        for hypothesis in result.hypotheses
    )


# ======================================================================
# Cost Gating
# ======================================================================

def test_mht_max_cost_rejects_far_association():

    associator = make_associator(
        max_cost=1.0,
        miss_cost=2.0,
        clutter_cost=2.0,
    )

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        100.0,
        0.0,
        0.0,
    )

    result = associator.associate(
        tracks=[track],
        detections=[detection],
    )

    assert len(result.hypotheses) > 0

    for hypothesis in result.hypotheses:

        for track_index, detection_index in hypothesis.matches:

            assert (
                result.cost_matrix[
                    track_index,
                    detection_index,
                ]
                <= 1.0
            )


# ======================================================================
# Detection Without Position
# ======================================================================

def test_mht_handles_detection_without_position():

    associator = make_associator()

    detection = DetectionResult(
        timestamp=Timestamp.now(),
        sensor=make_sensor(),
        class_id=0,
        class_name="drone",
        confidence=0.9,
        position=None,
    )

    result = associator.associate(
        tracks=[
            make_track(
                0.0,
                0.0,
                0.0,
            ),
        ],
        detections=[
            detection,
        ],
    )

    assert isinstance(
        result,
        MHTResult,
    )

    assert len(
        result.hypotheses
    ) > 0


# ======================================================================
# Track Identity Preservation
# ======================================================================

def test_mht_does_not_modify_track_identity():

    associator = make_associator()

    track = make_track(
        10.0,
        20.0,
        30.0,
    )

    original_id = track.track_id

    associator.associate(
        tracks=[track],
        detections=[
            make_detection(
                10.1,
                20.1,
                30.1,
            ),
        ],
    )

    assert track.track_id == original_id


# ======================================================================
# Track State Preservation
# ======================================================================

def test_mht_does_not_modify_track_state():

    associator = make_associator()

    track = make_track(
        10.0,
        20.0,
        30.0,
        vx=1.0,
        vy=2.0,
        vz=3.0,
    )

    original_state = track.state_vector.state.copy()
    original_covariance = (
        track.state_vector.covariance.copy()
    )

    associator.associate(
        tracks=[track],
        detections=[
            make_detection(
                10.5,
                20.5,
                30.5,
            ),
        ],
    )

    np.testing.assert_array_equal(
        track.state_vector.state,
        original_state,
    )

    np.testing.assert_array_equal(
        track.state_vector.covariance,
        original_covariance,
    )


# ======================================================================
# Repeatability
# ======================================================================

def test_mht_result_is_repeatable():

    associator = make_associator()

    tracks = [
        make_track(0.0, 0.0, 0.0),
        make_track(10.0, 0.0, 0.0),
    ]

    detections = [
        make_detection(0.2, 0.0, 0.0),
        make_detection(10.2, 0.0, 0.0),
    ]

    result_1 = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    result_2 = associator.associate(
        tracks=tracks,
        detections=detections,
    )

    np.testing.assert_allclose(
        result_1.cost_matrix,
        result_2.cost_matrix,
    )

    assert len(
        result_1.hypotheses
    ) == len(
        result_2.hypotheses
    )

    for h1, h2 in zip(
        result_1.hypotheses,
        result_2.hypotheses,
    ):

        assert h1.matches == h2.matches

        assert (
            h1.unmatched_tracks
            == h2.unmatched_tracks
        )

        assert (
            h1.unmatched_detections
            == h2.unmatched_detections
        )

        assert np.isclose(
            h1.total_cost,
            h2.total_cost,
        )

        assert np.isclose(
            h1.score,
            h2.score,
        )