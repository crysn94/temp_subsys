"""
Tests for Track Prediction.

TrackPrediction is responsible for generating future predicted states
from the current Track state.

Responsibilities tested here:

- construction
- prediction of a single track
- prediction using positive dt
- zero-dt behavior
- position prediction
- velocity preservation
- covariance propagation
- multiple-step prediction
- prediction without modifying the original Track
- deterministic behavior
- invalid input handling

TrackPrediction does NOT own:

- data association
- track creation
- track removal
- lifecycle transitions
- threat assessment
- sensor fusion
- measurement updates

The Track remains the canonical owner of the current state.

The StateVector remains the canonical owner of the dynamic state and
prediction operation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.timestamps import Timestamp

from tracking.models.state_vector import StateVector
from tracking.models.track import Track
from tracking.prediction.track_prediction import TrackPrediction


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
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> Track:

    return Track(
        track_id=track_id,
        state_vector=make_state_vector(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        ),
    )


# ======================================================================
# Construction
# ======================================================================


def test_track_prediction_can_be_constructed():

    predictor = TrackPrediction()

    assert predictor is not None


# ======================================================================
# Single Track Prediction
# ======================================================================


def test_predict_single_track():

    predictor = TrackPrediction()

    track = make_track(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=20.0,
        vz=5.0,
    )

    predicted = predictor.predict(
        track,
        dt=1.0,
    )

    assert predicted is not None


def test_prediction_returns_state_vector():

    predictor = TrackPrediction()

    track = make_track(
        vx=10.0,
    )

    predicted = predictor.predict(
        track,
        dt=1.0,
    )

    assert isinstance(
        predicted,
        StateVector,
    )


# ======================================================================
# Position Prediction
# ======================================================================


def test_prediction_updates_position():

    predictor = TrackPrediction()

    track = make_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=2.0,
        vz=-1.0,
    )

    predicted = predictor.predict(
        track,
        dt=2.0,
    )

    assert predicted.position.x == pytest.approx(20.0)
    assert predicted.position.y == pytest.approx(24.0)
    assert predicted.position.z == pytest.approx(28.0)


def test_prediction_with_zero_velocity_preserves_position():

    predictor = TrackPrediction()

    track = make_track(
        x=10.0,
        y=20.0,
        z=30.0,
    )

    predicted = predictor.predict(
        track,
        dt=5.0,
    )

    assert predicted.position.x == pytest.approx(10.0)
    assert predicted.position.y == pytest.approx(20.0)
    assert predicted.position.z == pytest.approx(30.0)


# ======================================================================
# Velocity Prediction
# ======================================================================


def test_constant_velocity_is_preserved():

    predictor = TrackPrediction()

    track = make_track(
        vx=10.0,
        vy=-5.0,
        vz=2.0,
    )

    predicted = predictor.predict(
        track,
        dt=3.0,
    )

    assert predicted.velocity.x == pytest.approx(10.0)
    assert predicted.velocity.y == pytest.approx(-5.0)
    assert predicted.velocity.z == pytest.approx(2.0)


# ======================================================================
# Zero Delta Time
# ======================================================================


def test_zero_dt_returns_same_state_values():

    predictor = TrackPrediction()

    track = make_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
    )

    predicted = predictor.predict(
        track,
        dt=0.0,
    )

    assert predicted.position.x == pytest.approx(10.0)
    assert predicted.position.y == pytest.approx(20.0)
    assert predicted.position.z == pytest.approx(30.0)

    assert predicted.velocity.x == pytest.approx(4.0)
    assert predicted.velocity.y == pytest.approx(5.0)
    assert predicted.velocity.z == pytest.approx(6.0)


# ======================================================================
# Negative Delta Time
# ======================================================================


def test_negative_dt_is_rejected():

    predictor = TrackPrediction()

    track = make_track()

    with pytest.raises(ValueError):

        predictor.predict(
            track,
            dt=-1.0,
        )


# ======================================================================
# Covariance
# ======================================================================


def test_prediction_preserves_valid_covariance_shape():

    predictor = TrackPrediction()

    track = make_track()

    predicted = predictor.predict(
        track,
        dt=1.0,
    )

    assert predicted.covariance.shape == (
        6,
        6,
    )


def test_prediction_covariance_is_finite():

    predictor = TrackPrediction()

    track = make_track()

    predicted = predictor.predict(
        track,
        dt=1.0,
    )

    assert np.all(
        np.isfinite(
            predicted.covariance
        )
    )


# ======================================================================
# Original Track Protection
# ======================================================================


def test_prediction_does_not_modify_original_track():

    predictor = TrackPrediction()

    track = make_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=2.0,
        vz=1.0,
    )

    original_position = (
        track.state_vector.position.x,
        track.state_vector.position.y,
        track.state_vector.position.z,
    )

    predictor.predict(
        track,
        dt=2.0,
    )

    current_position = (
        track.state_vector.position.x,
        track.state_vector.position.y,
        track.state_vector.position.z,
    )

    assert current_position == pytest.approx(
        original_position
    )


def test_prediction_does_not_modify_original_velocity():

    predictor = TrackPrediction()

    track = make_track(
        vx=5.0,
        vy=2.0,
        vz=1.0,
    )

    original_velocity = (
        track.state_vector.velocity.x,
        track.state_vector.velocity.y,
        track.state_vector.velocity.z,
    )

    predictor.predict(
        track,
        dt=2.0,
    )

    current_velocity = (
        track.state_vector.velocity.x,
        track.state_vector.velocity.y,
        track.state_vector.velocity.z,
    )

    assert current_velocity == pytest.approx(
        original_velocity
    )


# ======================================================================
# Multiple Prediction Steps
# ======================================================================


def test_multiple_prediction_steps_are_supported():

    predictor = TrackPrediction()

    track = make_track(
        x=0.0,
        vx=10.0,
    )

    predicted_1 = predictor.predict(
        track,
        dt=1.0,
    )

    predicted_2 = predictor.predict(
        track,
        dt=2.0,
    )

    assert predicted_1.position.x == pytest.approx(10.0)
    assert predicted_2.position.x == pytest.approx(20.0)


# ======================================================================
# Process Noise
# ======================================================================


def test_prediction_accepts_process_noise():

    predictor = TrackPrediction()

    track = make_track()

    process_noise = np.eye(6) * 0.1

    predicted = predictor.predict(
        track,
        dt=1.0,
        process_noise=process_noise,
    )

    assert isinstance(
        predicted,
        StateVector,
    )


# ======================================================================
# Track Validation
# ======================================================================


def test_prediction_requires_track():

    predictor = TrackPrediction()

    with pytest.raises(TypeError):

        predictor.predict(
            None,
            dt=1.0,
        )


def test_prediction_rejects_invalid_object():

    predictor = TrackPrediction()

    with pytest.raises(TypeError):

        predictor.predict(
            "not-a-track",
            dt=1.0,
        )


# ======================================================================
# Delta Time Validation
# ======================================================================


def test_non_numeric_dt_is_rejected():

    predictor = TrackPrediction()

    track = make_track()

    with pytest.raises((TypeError, ValueError)):

        predictor.predict(
            track,
            dt="1.0",
        )


def test_nonfinite_dt_is_rejected():

    predictor = TrackPrediction()

    track = make_track()

    with pytest.raises(ValueError):

        predictor.predict(
            track,
            dt=float("nan"),
        )


def test_infinite_dt_is_rejected():

    predictor = TrackPrediction()

    track = make_track()

    with pytest.raises(ValueError):

        predictor.predict(
            track,
            dt=float("inf"),
        )


# ======================================================================
# Repeatability
# ======================================================================


def test_prediction_is_repeatable():

    predictor = TrackPrediction()

    track = make_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=-2.0,
        vz=1.0,
    )

    predicted_1 = predictor.predict(
        track,
        dt=2.0,
    )

    predicted_2 = predictor.predict(
        track,
        dt=2.0,
    )

    assert predicted_1.position.x == pytest.approx(
        predicted_2.position.x
    )

    assert predicted_1.position.y == pytest.approx(
        predicted_2.position.y
    )

    assert predicted_1.position.z == pytest.approx(
        predicted_2.position.z
    )

    assert predicted_1.velocity.x == pytest.approx(
        predicted_2.velocity.x
    )

    assert predicted_1.velocity.y == pytest.approx(
        predicted_2.velocity.y
    )

    assert predicted_1.velocity.z == pytest.approx(
        predicted_2.velocity.z
    )


# ======================================================================
# Track Identity
# ======================================================================


def test_prediction_does_not_change_track_identity():

    predictor = TrackPrediction()

    track = make_track(
        track_id="TRACK-123",
    )

    predictor.predict(
        track,
        dt=1.0,
    )

    assert track.track_id == "TRACK-123"


# ======================================================================
# Representation
# ======================================================================


def test_track_prediction_repr():

    predictor = TrackPrediction()

    representation = repr(predictor)

    assert "TrackPrediction" in representation