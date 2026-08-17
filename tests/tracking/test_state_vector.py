"""
========================================================================
StateVector Tests
========================================================================

Tests for the canonical 3D constant-velocity StateVector.

State:

    [x, y, z, vx, vy, vz]

========================================================================
"""

import numpy as np
import pytest

from core.geometry.point import Point3D
from core.geometry.velocity import Velocity3D
from core.timestamps import Timestamp

from tracking.models.state_vector import StateVector


# ======================================================================
# Creation
# ======================================================================

def test_state_vector_creation():

    state = np.array([
        10.0,
        20.0,
        30.0,
        1.0,
        2.0,
        3.0,
    ])

    covariance = np.eye(6)

    timestamp = Timestamp.now()

    sv = StateVector(
        state=state,
        covariance=covariance,
        timestamp=timestamp,
    )

    assert sv.x == 10.0
    assert sv.y == 20.0
    assert sv.z == 30.0

    assert sv.vx == 1.0
    assert sv.vy == 2.0
    assert sv.vz == 3.0

    assert sv.timestamp == timestamp


# ======================================================================
# Position
# ======================================================================

def test_state_vector_position():

    sv = StateVector.from_components(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=1.0,
        vy=2.0,
        vz=3.0,
    )

    assert isinstance(
        sv.position,
        Point3D,
    )

    assert sv.position.x == 10.0
    assert sv.position.y == 20.0
    assert sv.position.z == 30.0


# ======================================================================
# Velocity
# ======================================================================

def test_state_vector_velocity():

    sv = StateVector.from_components(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=1.0,
        vy=2.0,
        vz=3.0,
    )

    assert isinstance(
        sv.velocity,
        Velocity3D,
    )

    assert sv.velocity.vx == 1.0
    assert sv.velocity.vy == 2.0
    assert sv.velocity.vz == 3.0


# ======================================================================
# Speed
# ======================================================================

def test_state_vector_speed():

    sv = StateVector.from_components(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=3.0,
        vy=4.0,
        vz=12.0,
    )

    assert sv.speed == pytest.approx(13.0)


# ======================================================================
# State prediction / representation
# ======================================================================

def test_state_vector_prediction():

    state = np.array([
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
    ])

    covariance = np.eye(6)

    timestamp = Timestamp.now()

    sv = StateVector(
        state=state,
        covariance=covariance,
        timestamp=timestamp,
    )

    assert sv.position.x == 0.0
    assert sv.position.y == 0.0
    assert sv.position.z == 0.0

    assert sv.velocity.vx == 10.0
    assert sv.velocity.vy == 20.0
    assert sv.velocity.vz == 30.0


# ======================================================================
# Array conversion
# ======================================================================

def test_state_vector_to_array():

    state = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    covariance = np.eye(6)

    sv = StateVector(
        state=state,
        covariance=covariance,
        timestamp=Timestamp.now(),
    )

    result = sv.to_array()

    assert isinstance(
        result,
        np.ndarray,
    )

    assert result.shape == (6,)

    np.testing.assert_array_equal(
        result,
        state,
    )


# ======================================================================
# Copy
# ======================================================================

def test_state_vector_copy():

    sv = StateVector.from_components(
        x=1.0,
        y=2.0,
        z=3.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
    )

    copied = sv.copy()

    assert copied is not sv

    np.testing.assert_array_equal(
        copied.state,
        sv.state,
    )

    np.testing.assert_array_equal(
        copied.covariance,
        sv.covariance,
    )

    assert copied.timestamp == sv.timestamp


# ======================================================================
# Factory from array
# ======================================================================

def test_state_vector_from_array():

    state = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    covariance = np.eye(6)

    timestamp = Timestamp.now()

    sv = StateVector.from_array(
        state=state,
        covariance=covariance,
        timestamp=timestamp,
    )

    np.testing.assert_array_equal(
        sv.state,
        state,
    )

    np.testing.assert_array_equal(
        sv.covariance,
        covariance,
    )

    assert sv.timestamp == timestamp


# ======================================================================
# Invalid state dimension
# ======================================================================

def test_state_vector_invalid_dimension():

    state = np.zeros(5)

    covariance = np.eye(5)

    with pytest.raises(ValueError):

        StateVector(
            state=state,
            covariance=covariance,
            timestamp=Timestamp.now(),
        )


# ======================================================================
# Invalid covariance dimension
# ======================================================================

def test_state_vector_invalid_covariance():

    state = np.zeros(6)

    covariance = np.eye(5)

    with pytest.raises(ValueError):

        StateVector(
            state=state,
            covariance=covariance,
            timestamp=Timestamp.now(),
        )


# ======================================================================
# Non-finite state
# ======================================================================

def test_state_vector_nonfinite_state():

    state = np.array([
        1.0,
        2.0,
        3.0,
        np.nan,
        5.0,
        6.0,
    ])

    covariance = np.eye(6)

    with pytest.raises(ValueError):

        StateVector(
            state=state,
            covariance=covariance,
            timestamp=Timestamp.now(),
        )


# ======================================================================
# Serialization
# ======================================================================

def test_state_vector_as_dict():

    sv = StateVector.from_components(
        x=1.0,
        y=2.0,
        z=3.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
    )

    result = sv.as_dict()

    assert "state" in result
    assert "position" in result
    assert "velocity" in result
    assert "covariance" in result
    assert "timestamp" in result

    assert result["position"]["x"] == 1.0
    assert result["position"]["y"] == 2.0
    assert result["position"]["z"] == 3.0

    assert result["velocity"]["vx"] == 4.0
    assert result["velocity"]["vy"] == 5.0
    assert result["velocity"]["vz"] == 6.0