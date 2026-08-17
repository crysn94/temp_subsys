from __future__ import annotations

import numpy as np
import pytest

from tracking.association.base_associator import (
    AssociationResult,
    BaseAssociator,
)
from core.detection_result import DetectionResult
from core.sensor_identifier import (
    SensorIdentifier,
    SensorCategory,
)
from tracking.models.track import Track


# ======================================================================
# Test Helpers
# ======================================================================


class DummyAssociator(BaseAssociator):
    """Minimal concrete associator for testing BaseAssociator."""

    def __init__(
        self,
        result: AssociationResult | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._result = result

    def _associate(
        self,
        tracks,
        detections,
    ) -> AssociationResult:

        if self._result is not None:
            return self._result

        return AssociationResult(
            matches=[],
        )


def make_track():
    """
    Create a minimal Track using the project's current Track API.

    Replace this helper only if the Track constructor changes.
    """

    # This intentionally uses the existing project Track construction
    # pattern. If your current Track requires additional fields,
    # adapt only this helper.
    return Track(
        track_id="test-track",
    )


def make_detection(
    detection_id: str = "test-detection",
):
    """
    Create a minimal canonical DetectionResult for tests.
    """

    sensor = SensorIdentifier(
        sensor_id="test-sensor",
        name="Test Sensor",
        category=SensorCategory.EO,
    )

    return DetectionResult(
        detection_id=detection_id,
        sensor=sensor,
        class_id=0,
        class_name="drone",
        confidence=0.95,
    )


# ======================================================================
# AssociationResult
# ======================================================================


def test_association_result_can_be_constructed():

    result = AssociationResult()

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []
    assert result.costs == {}
    assert result.metadata == {}


def test_association_result_properties():

    result = AssociationResult(
        matches=[
            (0, 0),
            (1, 1),
        ],
        unmatched_tracks=[2],
        unmatched_detections=[2, 3],
    )

    assert result.num_matches == 2
    assert result.num_unmatched_tracks == 1
    assert result.num_unmatched_detections == 2
    assert result.has_matches is True


def test_association_result_has_no_matches():

    result = AssociationResult()

    assert result.num_matches == 0
    assert result.has_matches is False


def test_association_result_as_dict():

    result = AssociationResult(
        matches=[(0, 1)],
        unmatched_tracks=[1],
        unmatched_detections=[0],
        costs={
            (0, 1): 2.5,
        },
        metadata={
            "algorithm": "test",
        },
    )

    data = result.as_dict()

    assert data["matches"][0]["track_index"] == 0
    assert data["matches"][0]["detection_index"] == 1
    assert data["matches"][0]["cost"] == 2.5

    assert data["unmatched_tracks"] == [1]
    assert data["unmatched_detections"] == [0]

    assert data["metadata"]["algorithm"] == "test"


# ======================================================================
# BaseAssociator Construction
# ======================================================================


def test_base_associator_is_abstract():

    with pytest.raises(TypeError):
        BaseAssociator()


def test_dummy_associator_can_be_constructed():

    associator = DummyAssociator()

    assert associator is not None


def test_gating_threshold_is_stored():

    associator = DummyAssociator(
        gating_threshold=5.0,
    )

    assert associator.gating_threshold == 5.0


def test_negative_gating_threshold_is_rejected():

    with pytest.raises(ValueError):

        DummyAssociator(
            gating_threshold=-1.0,
        )


def test_zero_gating_threshold_is_rejected():

    with pytest.raises(ValueError):

        DummyAssociator(
            gating_threshold=0.0,
        )


def test_none_gating_threshold_is_allowed():

    associator = DummyAssociator(
        gating_threshold=None,
    )

    assert associator.gating_threshold is None


# ======================================================================
# Empty Inputs
# ======================================================================


def test_empty_tracks():

    associator = DummyAssociator()

    result = associator.associate(
        [],
        [],
    )

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_empty_tracks_with_detections():

    detection = make_detection()

    associator = DummyAssociator()

    result = associator.associate(
        [],
        [detection],
    )

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == [0]


def test_tracks_with_empty_detections():

    track = make_track()

    associator = DummyAssociator()

    result = associator.associate(
        [track],
        [],
    )

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == []


# ======================================================================
# Input Validation
# ======================================================================


def test_none_tracks_are_rejected():

    associator = DummyAssociator()

    with pytest.raises(ValueError):

        associator.associate(
            None,
            [],
        )


def test_none_detections_are_rejected():

    associator = DummyAssociator()

    with pytest.raises(ValueError):

        associator.associate(
            [],
            None,
        )


def test_invalid_track_type_is_rejected():

    associator = DummyAssociator()

    detection = make_detection()

    with pytest.raises(TypeError):

        associator.associate(
            ["invalid"],
            [detection],
        )


def test_invalid_detection_type_is_rejected():

    associator = DummyAssociator()

    track = make_track()

    with pytest.raises(TypeError):

        associator.associate(
            [track],
            ["invalid"],
        )


# ======================================================================
# Result Normalization
# ======================================================================


def test_invalid_result_type_is_rejected():

    class InvalidAssociator(BaseAssociator):

        def _associate(
            self,
            tracks,
            detections,
        ):
            return "invalid"

    associator = InvalidAssociator()

    track = make_track()
    detection = make_detection()

    with pytest.raises(TypeError):

        associator.associate(
            [track],
            [detection],
        )


def test_out_of_range_matches_are_removed():

    result = AssociationResult(
        matches=[
            (0, 0),
            (100, 100),
        ],
    )

    associator = DummyAssociator(
        result=result,
    )

    track = make_track()
    detection = make_detection()

    normalized = associator.associate(
        [track],
        [detection],
    )

    assert normalized.matches == [
        (0, 0),
    ]


def test_duplicate_track_matches_are_removed():

    result = AssociationResult(
        matches=[
            (0, 0),
            (0, 1),
        ],
    )

    associator = DummyAssociator(
        result=result,
    )

    tracks = [
        make_track(),
    ]

    detections = [
        make_detection(),
        make_detection(),
    ]

    normalized = associator.associate(
        tracks,
        detections,
    )

    assert normalized.matches == [
        (0, 0),
    ]


def test_duplicate_detection_matches_are_removed():

    result = AssociationResult(
        matches=[
            (0, 0),
            (1, 0),
        ],
    )

    associator = DummyAssociator(
        result=result,
    )

    tracks = [
        make_track(),
        make_track(),
    ]

    detections = [
        make_detection(),
    ]

    normalized = associator.associate(
        tracks,
        detections,
    )

    assert normalized.matches == [
        (0, 0),
    ]


def test_unmatched_indices_are_recalculated():

    result = AssociationResult(
        matches=[
            (0, 0),
        ],
        unmatched_tracks=[
            999,
        ],
        unmatched_detections=[
            999,
        ],
    )

    associator = DummyAssociator(
        result=result,
    )

    tracks = [
        make_track(),
        make_track(),
    ]

    detections = [
        make_detection(),
        make_detection(),
    ]

    normalized = associator.associate(
        tracks,
        detections,
    )

    assert normalized.matches == [
        (0, 0),
    ]

    assert normalized.unmatched_tracks == [
        1,
    ]

    assert normalized.unmatched_detections == [
        1,
    ]


def test_valid_costs_are_preserved():

    result = AssociationResult(
        matches=[
            (0, 0),
        ],
        costs={
            (0, 0): 4.25,
        },
    )

    associator = DummyAssociator(
        result=result,
    )

    normalized = associator.associate(
        [make_track()],
        [make_detection()],
    )

    assert normalized.costs == {
        (0, 0): 4.25,
    }


def test_invalid_costs_are_removed():

    result = AssociationResult(
        matches=[
            (0, 0),
        ],
        costs={
            (0, 0): "invalid",
        },
    )

    associator = DummyAssociator(
        result=result,
    )

    normalized = associator.associate(
        [make_track()],
        [make_detection()],
    )

    assert normalized.costs == {}


# ======================================================================
# Properties
# ======================================================================


def test_name():

    associator = DummyAssociator()

    assert associator.name == "DummyAssociator"


def test_metric_name_without_builder():

    associator = DummyAssociator()

    assert associator.metric_name is None


def test_metric_name_with_builder():

    class DummyBuilder:

        metric_name = "EuclideanMetric"

    associator = DummyAssociator(
        cost_matrix_builder=DummyBuilder(),
    )

    assert associator.metric_name == "EuclideanMetric"


def test_repr():

    associator = DummyAssociator()

    representation = repr(associator)

    assert "DummyAssociator" in representation
    assert "gating_threshold" in representation