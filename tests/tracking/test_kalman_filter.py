"""
========================================================================
Kalman Filter Tests
========================================================================

Tests for the canonical 3D Constant-Acceleration Kalman Filter.

State model:

    [x, y, z,
     vx, vy, vz,
     ax, ay, az]

Measurement model:

    [x, y, z]

The tests validate:

    • KalmanFilter construction
    • StateCA prediction
    • Constant-acceleration motion
    • Covariance prediction
    • Measurement update
    • DetectionResult update
    • Innovation
    • Innovation covariance
    • Mahalanobis distance
    • Negative dt rejection
    • Invalid state rejection
    • Reset/copy behavior
    • Process-noise generation
    • Numerical stability
"""

import numpy as np
import pytest

from core.geometry.point import Point3D
from core.detection_result import DetectionResult
from core.sensor_identifier import (
    SensorCategory,
    SensorIdentifier,
)

from tracking.filters.kalman_filter import (
    KalmanFilter,
)

from tracking.models.state_ca import (
    StateCA,
)


# ======================================================================
# Helpers
# ======================================================================


def make_sensor() -> SensorIdentifier:

    return SensorIdentifier(
        sensor_id="TEST_SENSOR",
        name="Test Sensor",
        category=SensorCategory.EO,
    )


# ----------------------------------------------------------------------


def make_detection(
    x: float,
    y: float,
    z: float,
) -> DetectionResult:

    return DetectionResult(
        sensor=make_sensor(),
        class_id=1,
        class_name="Drone",
        confidence=0.95,
        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),
    )


# ----------------------------------------------------------------------


def make_state(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 0.0,
    covariance_scale: float = 1.0,
) -> StateCA:

    state = np.array(
        [
            x,
            y,
            z,
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
        ],
        dtype=float,
    )

    covariance = (
        np.eye(
            9,
            dtype=float,
        )
        * covariance_scale
    )

    return StateCA(
        state=state,
        covariance=covariance,
    )


# ======================================================================
# Initialization
# ======================================================================


def test_kalman_initialization():

    kf = KalmanFilter()

    assert kf.measurement_variance == pytest.approx(1.0)

    assert kf.jerk_variance == pytest.approx(1.0)

    assert kf.min_covariance == pytest.approx(1e-9)


# ======================================================================
# Configuration Validation
# ======================================================================


def test_negative_measurement_variance():

    with pytest.raises(ValueError):

        KalmanFilter(
            measurement_variance=-1.0,
        )


# ----------------------------------------------------------------------


def test_negative_jerk_variance():

    with pytest.raises(ValueError):

        KalmanFilter(
            jerk_variance=-1.0,
        )


# ----------------------------------------------------------------------


def test_invalid_min_covariance():

    with pytest.raises(ValueError):

        KalmanFilter(
            min_covariance=0.0,
        )


# ======================================================================
# Measurement Matrix
# ======================================================================


def test_measurement_matrix():

    kf = KalmanFilter()

    H = kf.measurement_matrix()

    assert H.shape == (
        3,
        9,
    )

    assert np.array_equal(
        H,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


# ======================================================================
# Measurement Noise
# ======================================================================


def test_measurement_noise_matrix():

    kf = KalmanFilter(
        measurement_variance=4.0,
    )

    R = kf.measurement_noise_matrix()

    assert R.shape == (
        3,
        3,
    )

    assert np.allclose(
        R,
        np.eye(3) * 4.0,
    )


# ======================================================================
# Prediction
# ======================================================================


def test_kalman_prediction():

    kf = KalmanFilter(
        jerk_variance=0.0,
    )

    state = make_state(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=20.0,
        vz=30.0,
    )

    predicted = kf.predict(
        state,
        2.0,
    )

    assert isinstance(
        predicted,
        StateCA,
    )

    assert predicted.x == pytest.approx(20.0)

    assert predicted.y == pytest.approx(40.0)

    assert predicted.z == pytest.approx(60.0)

    assert predicted.vx == pytest.approx(10.0)

    assert predicted.vy == pytest.approx(20.0)

    assert predicted.vz == pytest.approx(30.0)

    assert predicted.ax == pytest.approx(0.0)

    assert predicted.ay == pytest.approx(0.0)

    assert predicted.az == pytest.approx(0.0)


# ======================================================================
# Constant Acceleration Prediction
# ======================================================================


def test_constant_acceleration_prediction():

    kf = KalmanFilter(
        jerk_variance=0.0,
    )

    state = make_state(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=20.0,
        vz=30.0,
        ax=2.0,
        ay=4.0,
        az=6.0,
    )

    predicted = kf.predict(
        state,
        2.0,
    )

    # x = x0 + vx*dt + 0.5*a*dt^2
    #
    # x = 0 + 10*2 + 0.5*2*4
    #   = 24

    assert predicted.x == pytest.approx(24.0)

    # y = 0 + 20*2 + 0.5*4*4
    #   = 48

    assert predicted.y == pytest.approx(48.0)

    # z = 0 + 30*2 + 0.5*6*4
    #   = 72

    assert predicted.z == pytest.approx(72.0)

    assert predicted.vx == pytest.approx(14.0)

    assert predicted.vy == pytest.approx(28.0)

    assert predicted.vz == pytest.approx(42.0)

    assert predicted.ax == pytest.approx(2.0)

    assert predicted.ay == pytest.approx(4.0)

    assert predicted.az == pytest.approx(6.0)


# ======================================================================
# Prediction Covariance
# ======================================================================


def test_kalman_prediction_covariance():

    kf = KalmanFilter()

    state = make_state(
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )

    predicted = kf.predict(
        state,
        1.0,
    )

    assert predicted.covariance.shape == (
        9,
        9,
    )

    assert np.allclose(
        predicted.covariance,
        predicted.covariance.T,
    )

    assert np.all(
        np.isfinite(
            predicted.covariance
        )
    )


# ======================================================================
# Prediction With Zero dt
# ======================================================================


def test_prediction_zero_dt():

    kf = KalmanFilter()

    state = make_state(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=1.0,
        vy=2.0,
        vz=3.0,
        ax=4.0,
        ay=5.0,
        az=6.0,
    )

    predicted = kf.predict(
        state,
        0.0,
    )

    assert np.allclose(
        predicted.state,
        state.state,
    )

    assert np.allclose(
        predicted.covariance,
        state.covariance,
    )


# ======================================================================
# Negative dt
# ======================================================================


def test_kalman_negative_dt():

    kf = KalmanFilter()

    state = make_state()

    with pytest.raises(ValueError):

        kf.predict(
            state,
            -1.0,
        )


# ======================================================================
# Invalid State Type
# ======================================================================


def test_kalman_invalid_state_type():

    kf = KalmanFilter()

    with pytest.raises(TypeError):

        kf.predict(
            None,
            1.0,
        )


# ======================================================================
# Measurement Update
# ======================================================================


def test_kalman_update():

    kf = KalmanFilter(
        measurement_variance=1.0,
        jerk_variance=1.0,
    )

    state = make_state(
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=20.0,
        vz=30.0,
    )

    predicted = kf.predict(
        state,
        1.0,
    )

    measurement = np.array(
        [
            11.0,
            21.0,
            31.0,
        ],
        dtype=float,
    )

    updated = kf.update(
        predicted,
        measurement,
    )

    assert isinstance(
        updated,
        StateCA,
    )

    assert np.all(
        np.isfinite(
            updated.state
        )
    )

    assert np.all(
        np.isfinite(
            updated.covariance
        )
    )


# ======================================================================
# Update From Detection
# ======================================================================


def test_kalman_update_from_detection():

    kf = KalmanFilter()

    state = make_state(
        x=10.0,
        y=20.0,
        z=30.0,
    )

    detection = make_detection(
        11.0,
        21.0,
        31.0,
    )

    updated = kf.update_from_detection(
        state=state,
        detection=detection,
    )

    assert isinstance(
        updated,
        StateCA,
    )

    assert np.all(
        np.isfinite(
            updated.state
        )
    )

    assert np.all(
        np.isfinite(
            updated.covariance
        )
    )


# ======================================================================
# Detection Without Position
# ======================================================================


def test_update_from_detection_without_position():

    kf = KalmanFilter()

    state = make_state()

    detection = DetectionResult(
        sensor=make_sensor(),
        class_id=1,
        class_name="Drone",
        confidence=0.95,
        position=None,
    )

    with pytest.raises(ValueError):

        kf.update_from_detection(
            state,
            detection,
        )


# ======================================================================
# Innovation
# ======================================================================


def test_innovation():

    kf = KalmanFilter()

    state = make_state(
        x=10.0,
        y=20.0,
        z=30.0,
    )

    measurement = np.array(
        [
            11.0,
            22.0,
            35.0,
        ],
        dtype=float,
    )

    innovation = kf.innovation(
        state,
        measurement,
    )

    assert innovation.shape == (
        3,
    )

    assert np.allclose(
        innovation,
        np.array(
            [
                1.0,
                2.0,
                5.0,
            ]
        ),
    )


# ======================================================================
# Innovation Covariance
# ======================================================================


def test_innovation_covariance():

    kf = KalmanFilter(
        measurement_variance=2.0,
    )

    state = make_state(
        covariance_scale=3.0,
    )

    S = kf.innovation_covariance(
        state,
    )

    assert S.shape == (
        3,
        3,
    )

    assert np.allclose(
        S,
        np.eye(3) * 5.0,
    )


# ======================================================================
# Mahalanobis Distance
# ======================================================================


def test_mahalanobis_distance():

    kf = KalmanFilter(
        measurement_variance=1.0,
    )

    state = make_state(
        covariance_scale=1.0,
    )

    measurement = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    distance = kf.mahalanobis_distance(
        state,
        measurement,
    )

    assert distance == pytest.approx(
        1.0 / np.sqrt(2.0)
    )


# ======================================================================
# Process Noise
# ======================================================================


def test_constant_jerk_process_noise():

    Q = StateCA.constant_jerk_process_noise(
        dt=1.0,
        jerk_variance=1.0,
    )

    assert Q.shape == (
        9,
        9,
    )

    assert np.all(
        np.isfinite(Q)
    )

    assert np.allclose(
        Q,
        Q.T,
    )


# ======================================================================
# Process Noise Zero
# ======================================================================


def test_zero_jerk_process_noise():

    Q = StateCA.constant_jerk_process_noise(
        dt=1.0,
        jerk_variance=0.0,
    )

    assert np.allclose(
        Q,
        np.zeros(
            (
                9,
                9,
            )
        ),
    )


# ======================================================================
# Process Noise Invalid Parameters
# ======================================================================


def test_process_noise_negative_dt():

    with pytest.raises(ValueError):

        StateCA.constant_jerk_process_noise(
            dt=-1.0,
            jerk_variance=1.0,
        )


# ----------------------------------------------------------------------


def test_process_noise_negative_variance():

    with pytest.raises(ValueError):

        StateCA.constant_jerk_process_noise(
            dt=1.0,
            jerk_variance=-1.0,
        )


# ======================================================================
# Reset
# ======================================================================


def test_kalman_reset():

    kf = KalmanFilter()

    state = make_state(
        x=1.0,
        y=2.0,
        z=3.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
    )

    reset_state = kf.reset(
        state,
    )

    assert isinstance(
        reset_state,
        StateCA,
    )

    assert np.allclose(
        reset_state.state,
        state.state,
    )

    assert np.allclose(
        reset_state.covariance,
        state.covariance,
    )

    # Ensure defensive copy.

    assert reset_state is not state

    assert reset_state.state is not state.state

    assert (
        reset_state.covariance
        is not state.covariance
    )


# ======================================================================
# Reset Does Not Share Mutable Data
# ======================================================================


def test_reset_is_independent():

    kf = KalmanFilter()

    state = make_state(
        x=10.0,
    )

    reset_state = kf.reset(
        state,
    )

    reset_state.state[0] = 999.0

    assert state.state[0] == pytest.approx(
        10.0
    )


# ======================================================================
# Invalid Measurement
# ======================================================================


def test_invalid_measurement_shape():

    kf = KalmanFilter()

    state = make_state()

    with pytest.raises(ValueError):

        kf.update(
            state,
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
        )


# ----------------------------------------------------------------------


def test_nonfinite_measurement():

    kf = KalmanFilter()

    state = make_state()

    with pytest.raises(ValueError):

        kf.update(
            state,
            np.array(
                [
                    1.0,
                    np.nan,
                    3.0,
                ]
            ),
        )


# ======================================================================
# Invalid Measurement Noise
# ======================================================================


def test_invalid_measurement_noise_shape():

    kf = KalmanFilter()

    state = make_state()

    measurement = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    with pytest.raises(ValueError):

        kf.update(
            state,
            measurement,
            measurement_noise=np.eye(2),
        )


# ======================================================================
# Repeatability
# ======================================================================


def test_kalman_prediction_is_repeatable():

    kf = KalmanFilter()

    state = make_state(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=1.0,
        vy=2.0,
        vz=3.0,
        ax=0.5,
        ay=1.0,
        az=1.5,
    )

    predicted_1 = kf.predict(
        state,
        1.0,
    )

    predicted_2 = kf.predict(
        state,
        1.0,
    )

    assert np.allclose(
        predicted_1.state,
        predicted_2.state,
    )

    assert np.allclose(
        predicted_1.covariance,
        predicted_2.covariance,
    )