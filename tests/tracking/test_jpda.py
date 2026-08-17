"""
========================================================================
JPDA Association Tests
========================================================================

Tests the JPDA association layer using:

    Track
        +
    DetectionResult
        ↓
    CostMatrixBuilder
        ↓
    EuclideanMetric
        ↓
    JPDAAssociator

The tests use the canonical 3D StateVector:

    [x, y, z, vx, vy, vz]

========================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from core.detection_result import DetectionResult
from core.geometry.point import Point3D
from core.sensor_identifier import (
    SensorCategory,
    SensorIdentifier,
)

from tracking.association.cost_matrix import (
    CostMatrixBuilder,
)

from tracking.association.jpda import (
    JPDAAssociator,
)

from tracking.models.state_vector import (
    StateVector,
)

from tracking.models.track import (
    Track,
)

from tracking.similarity.euclidean import (
    EuclideanMetric,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sensor() -> SensorIdentifier:
    """
    Canonical test sensor.
    """

    return SensorIdentifier(
        sensor_id="TEST-E0-001",
        name="Test EO Sensor",
        category=SensorCategory.EO,
    )


@pytest.fixture
def metric() -> EuclideanMetric:
    """
    3D Euclidean association metric.
    """

    return EuclideanMetric()


@pytest.fixture
def cost_builder(
    metric: EuclideanMetric,
) -> CostMatrixBuilder:
    """
    Cost matrix builder using the Euclidean metric.
    """

    return CostMatrixBuilder(
        metric=metric
    )


@pytest.fixture
def jpda(
    cost_builder: CostMatrixBuilder,
) -> JPDAAssociator:
    """
    Construct JPDA with the required cost builder.
    """

    return JPDAAssociator(
        cost_builder=cost_builder
    )


# ======================================================================
# Helpers
# ======================================================================


def make_state(
    x: float,
    y: float,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> StateVector:
    """
    Create a canonical 3D StateVector.

    State ordering:

        [x, y, z, vx, vy, vz]
    """

    return StateVector.from_components(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        covariance=np.eye(6),
    )


def make_track(
    x: float,
    y: float,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> Track:
    """
    Create a Track with a 3D state.
    """

    return Track(
        state_vector=make_state(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )
    )


def make_detection(
    sensor: SensorIdentifier,
    x: float,
    y: float,
    z: float = 0.0,
    confidence: float = 0.95,
) -> DetectionResult:
    """
    Create a 3D DetectionResult.
    """

    return DetectionResult(
        sensor=sensor,
        class_id=1,
        class_name="drone",
        confidence=confidence,
        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),
    )


# ======================================================================
# Construction
# ======================================================================


def test_jpda_can_be_constructed(jpda):

    assert jpda is not None

    assert jpda.cost_builder is not None

    assert jpda.metric_name


# ======================================================================
# Empty Inputs
# ======================================================================


def test_jpda_empty_tracks(
    jpda,
    sensor,
):

    detections = [
        make_detection(
            sensor,
            10.0,
            10.0,
            5.0,
        )
    ]

    result = jpda.associate(
        [],
        detections,
    )

    assert result.probabilities.shape == (
        0,
        1,
    )

    assert result.missed_detection_probabilities.shape == (
        0,
    )

    assert result.matches == []

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == [0]


def test_jpda_empty_detections(
    jpda,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        )
    ]

    result = jpda.associate(
        tracks,
        [],
    )

    assert result.probabilities.shape == (
        1,
        0,
    )

    assert result.missed_detection_probabilities.shape == (
        1,
    )

    assert result.missed_detection_probabilities[0] == 1.0

    assert result.matches == []

    assert result.unmatched_tracks == [0]

    assert result.unmatched_detections == []


def test_jpda_empty_inputs(
    jpda,
):

    result = jpda.associate(
        [],
        [],
    )

    assert result.probabilities.shape == (
        0,
        0,
    )

    assert result.missed_detection_probabilities.shape == (
        0,
    )

    assert result.matches == []

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == []


# ======================================================================
# Single Association
# ======================================================================


def test_single_track_single_detection(
    jpda,
    sensor,
):

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = make_detection(
        sensor,
        0.1,
        0.1,
        0.1,
    )

    result = jpda.associate(
        [track],
        [detection],
    )

    assert result.probabilities.shape == (
        1,
        1,
    )

    probability = result.probability(
        0,
        0,
    )

    assert np.isfinite(
        probability
    )

    assert probability >= 0.0

    assert probability <= 1.0


# ======================================================================
# Distance Behaviour
# ======================================================================


def test_far_detection_has_lower_association_probability(
    jpda,
    sensor,
):

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    near_detection = make_detection(
        sensor,
        1.0,
        0.0,
        0.0,
    )

    far_detection = make_detection(
        sensor,
        100.0,
        0.0,
        0.0,
    )

    near_result = jpda.associate(
        [track],
        [near_detection],
    )

    far_result = jpda.associate(
        [track],
        [far_detection],
    )

    near_probability = (
        near_result.probability(0, 0)
    )

    far_probability = (
        far_result.probability(0, 0)
    )

    assert near_probability > far_probability


# ======================================================================
# Multiple Tracks / Detections
# ======================================================================


def test_multiple_tracks_multiple_detections(
    jpda,
    sensor,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        ),
        make_track(
            100.0,
            100.0,
            10.0,
        ),
    ]

    detections = [
        make_detection(
            sensor,
            1.0,
            1.0,
            0.0,
        ),
        make_detection(
            sensor,
            101.0,
            101.0,
            10.0,
        ),
    ]

    result = jpda.associate(
        tracks,
        detections,
    )

    assert result.probabilities.shape == (
        2,
        2,
    )

    assert np.all(
        np.isfinite(
            result.probabilities
        )
    )

    assert result.probabilities[0, 0] > result.probabilities[0, 1]

    assert result.probabilities[1, 1] > result.probabilities[1, 0]


# ======================================================================
# Probability Validation
# ======================================================================


def test_probability_values_are_finite(
    jpda,
    sensor,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        ),
        make_track(
            50.0,
            50.0,
            10.0,
        ),
    ]

    detections = [
        make_detection(
            sensor,
            1.0,
            1.0,
            1.0,
        ),
        make_detection(
            sensor,
            49.0,
            49.0,
            9.0,
        ),
    ]

    result = jpda.associate(
        tracks,
        detections,
    )

    assert np.all(
        np.isfinite(
            result.probabilities
        )
    )

    assert np.all(
        np.isfinite(
            result.missed_detection_probabilities
        )
    )


def test_probability_values_are_non_negative(
    jpda,
    sensor,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        )
    ]

    detections = [
        make_detection(
            sensor,
            1.0,
            1.0,
            1.0,
        ),
        make_detection(
            sensor,
            5.0,
            5.0,
            5.0,
        ),
    ]

    result = jpda.associate(
        tracks,
        detections,
    )

    assert np.all(
        result.probabilities >= 0.0
    )

    assert np.all(
        result.missed_detection_probabilities >= 0.0
    )


# ======================================================================
# Ambiguous Association
# ======================================================================


def test_jpda_handles_ambiguous_detections(
    jpda,
    sensor,
):

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detections = [
        make_detection(
            sensor,
            1.0,
            0.0,
            0.0,
        ),
        make_detection(
            sensor,
            1.1,
            0.0,
            0.0,
        ),
    ]

    result = jpda.associate(
        [track],
        detections,
    )

    assert result.probabilities.shape == (
        1,
        2,
    )

    p0 = result.probability(
        0,
        0,
    )

    p1 = result.probability(
        0,
        1,
    )

    assert p0 > 0.0

    assert p1 > 0.0

    assert np.isclose(
        p0 + p1 + result.missed_detection_probabilities[0],
        1.0,
    )


# ======================================================================
# Track Integrity
# ======================================================================


def test_jpda_does_not_modify_track_identity(
    jpda,
    sensor,
):

    track = make_track(
        10.0,
        20.0,
        30.0,
    )

    original_id = track.track_id

    detection = make_detection(
        sensor,
        10.1,
        20.1,
        30.1,
    )

    jpda.associate(
        [track],
        [detection],
    )

    assert track.track_id == original_id


def test_jpda_does_not_modify_track_state(
    jpda,
    sensor,
):

    track = make_track(
        10.0,
        20.0,
        30.0,
        1.0,
        2.0,
        3.0,
    )

    original_state = track.state_vector.state.copy()

    original_covariance = (
        track.state_vector.covariance.copy()
    )

    detection = make_detection(
        sensor,
        10.1,
        20.1,
        30.1,
    )

    jpda.associate(
        [track],
        [detection],
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
# Missing Position
# ======================================================================


def test_jpda_handles_detection_without_position(
    jpda,
    sensor,
):

    track = make_track(
        0.0,
        0.0,
        0.0,
    )

    detection = DetectionResult(
        sensor=sensor,
        class_id=1,
        class_name="drone",
        confidence=0.9,
        position=None,
    )

    result = jpda.associate(
        [track],
        [detection],
    )

    assert result.probabilities.shape == (
        1,
        1,
    )

    assert np.isfinite(
        result.probabilities[0, 0]
    )

    assert (
        result.probabilities[0, 0]
        == 0.0
    )

    assert (
        result.missed_detection_probabilities[0]
        == 1.0
    )


# ======================================================================
# Active Tracks
# ======================================================================


def test_jpda_can_process_active_tracks_only(
    jpda,
    sensor,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        ),
        make_track(
            100.0,
            100.0,
            100.0,
        ),
    ]

    active_tracks = [
        track
        for track in tracks
        if track.is_active
    ]

    detections = [
        make_detection(
            sensor,
            1.0,
            1.0,
            1.0,
        )
    ]

    result = jpda.associate(
        active_tracks,
        detections,
    )

    assert result.num_tracks == len(
        active_tracks
    )

    assert result.num_detections == 1


# ======================================================================
# Large Coordinates
# ======================================================================


def test_jpda_handles_large_coordinates(
    jpda,
    sensor,
):

    track = make_track(
        1_000_000.0,
        2_000_000.0,
        500_000.0,
    )

    detection = make_detection(
        sensor,
        1_000_001.0,
        2_000_001.0,
        500_001.0,
    )

    result = jpda.associate(
        [track],
        [detection],
    )

    assert np.all(
        np.isfinite(
            result.probabilities
        )
    )

    assert np.all(
        result.probabilities >= 0.0
    )


# ======================================================================
# Repeatability
# ======================================================================


def test_jpda_result_is_repeatable(
    jpda,
    sensor,
):

    tracks = [
        make_track(
            0.0,
            0.0,
            0.0,
        ),
        make_track(
            100.0,
            100.0,
            50.0,
        ),
    ]

    detections = [
        make_detection(
            sensor,
            1.0,
            1.0,
            1.0,
        ),
        make_detection(
            sensor,
            101.0,
            101.0,
            51.0,
        ),
    ]

    result1 = jpda.associate(
        tracks,
        detections,
    )

    result2 = jpda.associate(
        tracks,
        detections,
    )

    np.testing.assert_array_equal(
        result1.probabilities,
        result2.probabilities,
    )

    np.testing.assert_array_equal(
        result1.missed_detection_probabilities,
        result2.missed_detection_probabilities,
    )

    assert result1.matches == result2.matches