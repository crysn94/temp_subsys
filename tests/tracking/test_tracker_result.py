"""
Tests for TrackerResult.

TrackerResult is the public result object returned by
Tracker.update().
"""

from __future__ import annotations

from tracking.tracker import TrackerResult


# ======================================================================
# Construction
# ======================================================================


def test_tracker_result_can_be_constructed():

    result = TrackerResult()

    assert result is not None


# ======================================================================
# Default values
# ======================================================================


def test_default_collections_are_empty():

    result = TrackerResult()

    assert result.tracks == []
    assert result.active_tracks == []

    assert result.created_track_ids == []
    assert result.updated_track_ids == []
    assert result.missed_track_ids == []
    assert result.deleted_track_ids == []

    assert result.metadata == {}


def test_default_association_is_none():

    result = TrackerResult()

    assert result.association is None


# ======================================================================
# Counters
# ======================================================================


def test_default_counters_are_zero():

    result = TrackerResult()

    assert result.num_tracks == 0
    assert result.num_active_tracks == 0
    assert result.num_created_tracks == 0
    assert result.num_updated_tracks == 0
    assert result.num_missed_tracks == 0
    assert result.num_deleted_tracks == 0


def test_created_track_count():

    result = TrackerResult(
        created_track_ids=[
            "T0001",
            "T0002",
            "T0003",
        ]
    )

    assert result.num_created_tracks == 3


def test_updated_track_count():

    result = TrackerResult(
        updated_track_ids=[
            "T0001",
            "T0002",
        ]
    )

    assert result.num_updated_tracks == 2


def test_missed_track_count():

    result = TrackerResult(
        missed_track_ids=[
            "T0003",
            "T0004",
        ]
    )

    assert result.num_missed_tracks == 2


def test_deleted_track_count():

    result = TrackerResult(
        deleted_track_ids=[
            "T0005",
        ]
    )

    assert result.num_deleted_tracks == 1


def test_track_count():

    result = TrackerResult(
        tracks=[
            object(),
            object(),
            object(),
        ]
    )

    assert result.num_tracks == 3


def test_active_track_count():

    result = TrackerResult(
        active_tracks=[
            object(),
            object(),
        ]
    )

    assert result.num_active_tracks == 2


# ======================================================================
# Serialization
# ======================================================================


def test_as_dict_returns_dictionary():

    result = TrackerResult()

    output = result.as_dict()

    assert isinstance(output, dict)


def test_as_dict_contains_expected_keys():

    result = TrackerResult()

    output = result.as_dict()

    expected_keys = {
        "tracks",
        "active_tracks",
        "created_track_ids",
        "updated_track_ids",
        "missed_track_ids",
        "deleted_track_ids",
        "metadata",
    }

    assert set(output.keys()) == expected_keys


def test_as_dict_serializes_track_ids():

    result = TrackerResult(
        created_track_ids=["T0001"],
        updated_track_ids=["T0002"],
        missed_track_ids=["T0003"],
        deleted_track_ids=["T0004"],
        metadata={
            "num_detections_processed": 4,
        },
    )

    output = result.as_dict()

    assert output["created_track_ids"] == ["T0001"]
    assert output["updated_track_ids"] == ["T0002"]
    assert output["missed_track_ids"] == ["T0003"]
    assert output["deleted_track_ids"] == ["T0004"]

    assert output["metadata"] == {
        "num_detections_processed": 4,
    }


# ======================================================================
# Track serialization
# ======================================================================


def test_as_dict_serializes_tracks_by_id():

    class DummyTrack:

        def __init__(
            self,
            track_id: str,
        ):
            self.track_id = track_id

    tracks = [
        DummyTrack("T0001"),
        DummyTrack("T0002"),
    ]

    active_tracks = [
        DummyTrack("T0001"),
    ]

    result = TrackerResult(
        tracks=tracks,
        active_tracks=active_tracks,
    )

    output = result.as_dict()

    assert output["tracks"] == [
        "T0001",
        "T0002",
    ]

    assert output["active_tracks"] == [
        "T0001",
    ]


# ======================================================================
# Metadata isolation
# ======================================================================


def test_as_dict_does_not_expose_metadata_object():

    metadata = {
        "num_detections_processed": 5,
    }

    result = TrackerResult(
        metadata=metadata,
    )

    output = result.as_dict()

    output["metadata"][
        "num_detections_processed"
    ] = 999

    assert (
        result.metadata[
            "num_detections_processed"
        ]
        == 5
    )


# ======================================================================
# Counter consistency
# ======================================================================


def test_counter_consistency():

    result = TrackerResult(
        created_track_ids=[
            "T0001",
            "T0002",
        ],
        updated_track_ids=[
            "T0003",
        ],
        missed_track_ids=[
            "T0004",
            "T0005",
            "T0006",
        ],
        deleted_track_ids=[
            "T0007",
        ],
    )

    assert result.num_created_tracks == 2
    assert result.num_updated_tracks == 1
    assert result.num_missed_tracks == 3
    assert result.num_deleted_tracks == 1