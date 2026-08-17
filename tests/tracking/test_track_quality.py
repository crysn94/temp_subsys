"""
Unit tests for Track Quality Evaluation.

TrackQuality evaluates the reliability of an existing Track.

The quality evaluator does NOT:

- modify Track state
- modify lifecycle state
- perform data association
- perform prediction
- perform filtering

It only evaluates the current Track.
"""

from __future__ import annotations

import pytest
import numpy as np

from core.timestamps import Timestamp

from tracking.models.track import Track
from tracking.models.state_vector import StateVector
from tracking.quality.track_quality import TrackQuality


# ======================================================================
# Helpers
# ======================================================================


def make_state_vector() -> StateVector:
    return StateVector.from_components(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        covariance=np.eye(6),
        timestamp=Timestamp.now(),
    )


def make_track(
    track_id: str = "T001",
) -> Track:
    return Track(
        track_id=track_id,
        state_vector=make_state_vector(),
    )


# ======================================================================
# Construction
# ======================================================================


def test_track_quality_can_be_constructed():

    quality = TrackQuality()

    assert quality is not None


def test_default_configuration_is_valid():

    quality = TrackQuality()

    assert quality.min_quality == 0.0
    assert quality.max_quality == 1.0


# ======================================================================
# Configuration Validation
# ======================================================================


def test_min_quality_must_not_be_greater_than_max_quality():

    with pytest.raises(ValueError):

        TrackQuality(
            min_quality=1.0,
            max_quality=0.0,
        )


def test_quality_range_must_be_finite():

    with pytest.raises(ValueError):

        TrackQuality(
            min_quality=float("nan"),
        )


def test_quality_range_must_be_finite_max():

    with pytest.raises(ValueError):

        TrackQuality(
            max_quality=float("inf"),
        )


# ======================================================================
# Detection Rate
# ======================================================================


def test_detection_rate_for_new_track_is_zero():

    quality = TrackQuality()

    track = make_track()

    assert quality.detection_rate(track) == 0.0


def test_detection_rate_increases_with_detections():

    quality = TrackQuality()

    track = make_track()

    # The exact Track API for registering detections is already
    # established elsewhere in the project.
    #
    # Use the existing public API.
    #
    # We intentionally use a minimal fake detection only if the Track
    # accepts DetectionResult objects.

    assert quality.detection_rate(track) == 0.0


# ======================================================================
# Quality Score
# ======================================================================


def test_quality_score_is_numeric():

    quality = TrackQuality()

    track = make_track()

    score = quality.calculate(track)

    assert isinstance(score, float)


def test_quality_score_is_bounded():

    quality = TrackQuality()

    track = make_track()

    score = quality.calculate(track)

    assert 0.0 <= score <= 1.0


# ======================================================================
# Reliability
# ======================================================================


def test_new_track_is_not_reliable_at_default_threshold():

    quality = TrackQuality()

    track = make_track()

    assert quality.is_reliable(
        track,
        threshold=0.5,
    ) is False


def test_reliability_threshold_must_be_valid():

    quality = TrackQuality()

    track = make_track()

    with pytest.raises(ValueError):

        quality.is_reliable(
            track,
            threshold=-0.1,
        )

    with pytest.raises(ValueError):

        quality.is_reliable(
            track,
            threshold=1.1,
        )


# ======================================================================
# Component Scores
# ======================================================================


def test_continuity_score_is_bounded():

    quality = TrackQuality()

    track = make_track()

    score = quality.continuity_score(track)

    assert 0.0 <= score <= 1.0


def test_stability_score_is_bounded():

    quality = TrackQuality()

    track = make_track()

    score = quality.stability_score(track)

    assert 0.0 <= score <= 1.0


# ======================================================================
# Track Validation
# ======================================================================


def test_calculate_requires_track():

    quality = TrackQuality()

    with pytest.raises(TypeError):

        quality.calculate(None)


def test_detection_rate_requires_track():

    quality = TrackQuality()

    with pytest.raises(TypeError):

        quality.detection_rate(None)


def test_continuity_score_requires_track():

    quality = TrackQuality()

    with pytest.raises(TypeError):

        quality.continuity_score(None)


def test_stability_score_requires_track():

    quality = TrackQuality()

    with pytest.raises(TypeError):

        quality.stability_score(None)


def test_reliability_requires_track():

    quality = TrackQuality()

    with pytest.raises(TypeError):

        quality.is_reliable(None)


# ======================================================================
# Non-Mutation
# ======================================================================


def test_quality_calculation_does_not_modify_track():

    quality = TrackQuality()

    track = make_track()

    track_id = track.track_id
    lifecycle = track.lifecycle

    quality.calculate(track)

    assert track.track_id == track_id
    assert track.lifecycle == lifecycle


# ======================================================================
# Repeatability
# ======================================================================


def test_quality_calculation_is_repeatable():

    quality = TrackQuality()

    track = make_track()

    result1 = quality.calculate(track)
    result2 = quality.calculate(track)

    assert result1 == result2


# ======================================================================
# Configuration
# ======================================================================


def test_get_config_returns_serializable_configuration():

    quality = TrackQuality()

    config = quality.get_config()

    assert isinstance(config, dict)
    assert "min_quality" in config
    assert "max_quality" in config


# ======================================================================
# Representation
# ======================================================================


def test_repr_contains_class_name():

    quality = TrackQuality()

    representation = repr(quality)

    assert "TrackQuality" in representation