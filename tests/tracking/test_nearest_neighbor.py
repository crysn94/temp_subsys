"""
========================================================================
Nearest Neighbor Association Tests
========================================================================

Tests for:

    tracking.association.nearest_neighbor

The tests intentionally use a fake CostMatrixBuilder so that the
NearestNeighborAssociator is tested independently of:

    • Euclidean metric
    • Mahalanobis metric
    • IoU / GIoU
    • StateVector
    • DetectionResult geometry
    • Sensor implementations

The responsibility of this test module is specifically to verify the
association algorithm.

========================================================================
"""

from __future__ import annotations

import numpy as np

from tracking.association.nearest_neighbor import (
    AssociationResult,
    NearestNeighborAssociator,
)


# ======================================================================
# Test Helpers
# ======================================================================


class FakeCostBuilder:
    """
    Minimal CostMatrixBuilder replacement used for unit testing.

    The nearest-neighbor associator only requires:

        • build()
        • metric_name

    This keeps the tests independent of the actual metric
    implementation.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        metric_name: str = "FakeMetric",
    ) -> None:

        self.matrix = np.asarray(
            matrix,
            dtype=float,
        )

        self._metric_name = metric_name

    # ------------------------------------------------------------------

    def build(
        self,
        tracks,
        detections,
    ) -> np.ndarray:

        expected_shape = (
            len(tracks),
            len(detections),
        )

        if self.matrix.shape != expected_shape:
            raise ValueError(
                "Fake cost matrix shape does not match "
                f"tracks × detections: "
                f"{self.matrix.shape} != {expected_shape}"
            )

        return self.matrix.copy()

    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str:
        return self._metric_name


# ======================================================================
# Basic Association
# ======================================================================


def test_single_track_single_detection():

    builder = FakeCostBuilder(
        [[0.5]]
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = ["track_0"]

    detections = ["detection_0"]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == [
        (0, 0)
    ]

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == []

    assert result.num_matches == 1


# ======================================================================
# Global Greedy Matching
# ======================================================================


def test_global_greedy_matching():

    matrix = np.array(
        [
            [0.20, 0.90],
            [0.10, 0.80],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = [
        "track_0",
        "track_1",
    ]

    detections = [
        "detection_0",
        "detection_1",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    # Lowest cost is:
    #
    # track_1 -> detection_0 = 0.10
    #
    # Remaining valid pair:
    #
    # track_0 -> detection_1 = 0.90

    assert result.matches == [
        (1, 0),
        (0, 1),
    ]

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == []


# ======================================================================
# One-to-One Constraint
# ======================================================================


def test_one_to_one_assignment():

    matrix = np.array(
        [
            [0.10, 0.20],
            [0.15, 0.90],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = [
        "track_0",
        "track_1",
    ]

    detections = [
        "detection_0",
        "detection_1",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    # track_0 -> detection_0 is selected first.
    #
    # detection_0 cannot subsequently be reused.
    #
    # track_1 -> detection_1 is therefore selected.

    assert result.matches == [
        (0, 0),
        (1, 1),
    ]

    assert len(
        {track for track, _ in result.matches}
    ) == len(result.matches)

    assert len(
        {detection for _, detection in result.matches}
    ) == len(result.matches)


# ======================================================================
# Unmatched Track
# ======================================================================


def test_unmatched_track():

    matrix = np.array(
        [
            [0.10],
            [10.0],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = [
        "track_0",
        "track_1",
    ]

    detections = [
        "detection_0",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == [
        (0, 0)
    ]

    assert result.unmatched_tracks == [
        1
    ]

    assert result.unmatched_detections == []


# ======================================================================
# Unmatched Detection
# ======================================================================


def test_unmatched_detection():

    matrix = np.array(
        [
            [0.10, 10.0],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = [
        "track_0",
    ]

    detections = [
        "detection_0",
        "detection_1",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == [
        (0, 0)
    ]

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == [
        1
    ]


# ======================================================================
# Gating
# ======================================================================


def test_max_cost_gating():

    matrix = np.array(
        [
            [2.0],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=1.0,
    )

    tracks = [
        "track_0",
    ]

    detections = [
        "detection_0",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == []

    assert result.unmatched_tracks == [
        0
    ]

    assert result.unmatched_detections == [
        0
    ]


# ======================================================================
# Invalid Cost
# ======================================================================


def test_invalid_cost_is_not_associated():

    invalid_cost = 1e9

    matrix = np.array(
        [
            [invalid_cost],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=100.0,
        invalid_cost=invalid_cost,
    )

    tracks = [
        "track_0",
    ]

    detections = [
        "detection_0",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == []

    assert result.unmatched_tracks == [
        0
    ]

    assert result.unmatched_detections == [
        0
    ]


# ======================================================================
# Empty Tracks
# ======================================================================


def test_empty_tracks():

    builder = FakeCostBuilder(
        np.empty(
            (0, 2),
            dtype=float,
        )
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    tracks = []

    detections = [
        "detection_0",
        "detection_1",
    ]

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == []

    assert result.unmatched_tracks == []

    assert result.unmatched_detections == [
        0,
        1,
    ]

    assert result.cost_matrix.shape == (
        0,
        2,
    )


# ======================================================================
# Empty Detections
# ======================================================================


def test_empty_detections():

    builder = FakeCostBuilder(
        np.empty(
            (2, 0),
            dtype=float,
        )
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    tracks = [
        "track_0",
        "track_1",
    ]

    detections = []

    result = associator.associate(
        tracks,
        detections,
    )

    assert result.matches == []

    assert result.unmatched_tracks == [
        0,
        1,
    ]

    assert result.unmatched_detections == []

    assert result.cost_matrix.shape == (
        2,
        0,
    )


# ======================================================================
# Both Empty
# ======================================================================


def test_empty_tracks_and_detections():

    builder = FakeCostBuilder(
        np.empty(
            (0, 0),
            dtype=float,
        )
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

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
# Cost Matrix Preservation
# ======================================================================


def test_cost_matrix_is_preserved():

    matrix = np.array(
        [
            [0.10, 0.40],
            [0.20, 0.30],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    result = associator.associate(
        ["track_0", "track_1"],
        ["detection_0", "detection_1"],
    )

    np.testing.assert_array_equal(
        result.cost_matrix,
        matrix,
    )


# ======================================================================
# Match Cost
# ======================================================================


def test_get_match_cost():

    matrix = np.array(
        [
            [0.10, 0.80],
            [0.40, 0.20],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    result = associator.associate(
        ["track_0", "track_1"],
        ["detection_0", "detection_1"],
    )

    assert result.get_match_cost(
        0,
        0,
    ) == 0.10

    assert result.get_match_cost(
        1,
        1,
    ) == 0.20


# ======================================================================
# AssociationResult Serialization
# ======================================================================


def test_association_result_as_dict():

    matrix = np.array(
        [
            [0.10],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    result = associator.associate(
        ["track_0"],
        ["detection_0"],
    )

    data = result.as_dict()

    assert data["matches"][0][
        "track_index"
    ] == 0

    assert data["matches"][0][
        "detection_index"
    ] == 0

    assert data["matches"][0][
        "cost"
    ] == 0.10

    assert data["num_matches"] == 1


# ======================================================================
# Metric Name
# ======================================================================


def test_metric_name():

    builder = FakeCostBuilder(
        [[0.1]],
        metric_name="EuclideanMetric",
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    assert associator.metric_name == (
        "EuclideanMetric"
    )


# ======================================================================
# Configuration
# ======================================================================


def test_get_config():

    builder = FakeCostBuilder(
        [[0.1]],
        metric_name="MahalanobisMetric",
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=5.0,
        invalid_cost=1e9,
    )

    config = associator.get_config()

    assert config["algorithm"] == (
        "nearest_neighbor"
    )

    assert config["strategy"] == (
        "greedy_global"
    )

    assert config["metric"] == (
        "MahalanobisMetric"
    )

    assert config["max_cost"] == 5.0

    assert config["invalid_cost"] == 1e9


# ======================================================================
# Representation
# ======================================================================


def test_repr():

    builder = FakeCostBuilder(
        [[0.1]],
        metric_name="EuclideanMetric",
    )

    associator = NearestNeighborAssociator(
        cost_builder=builder,
        max_cost=10.0,
    )

    representation = repr(
        associator
    )

    assert (
        "NearestNeighborAssociator"
        in representation
    )

    assert (
        "EuclideanMetric"
        in representation
    )


# ======================================================================
# Alias
# ======================================================================


def test_associate_objects_alias():

    matrix = np.array(
        [
            [0.1],
        ]
    )

    builder = FakeCostBuilder(matrix)

    associator = NearestNeighborAssociator(
        cost_builder=builder,
    )

    tracks = [
        "track_0",
    ]

    detections = [
        "detection_0",
    ]

    result_1 = associator.associate(
        tracks,
        detections,
    )

    result_2 = associator.associate_objects(
        tracks,
        detections,
    )

    assert result_1.matches == result_2.matches

    assert (
        result_1.unmatched_tracks
        == result_2.unmatched_tracks
    )

    assert (
        result_1.unmatched_detections
        == result_2.unmatched_detections
    )