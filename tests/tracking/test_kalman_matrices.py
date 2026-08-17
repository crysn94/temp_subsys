import numpy as np

from tracking.math.kalman_matrices import (
    constant_velocity_transition_matrix,
    constant_velocity_process_noise,
    position_measurement_matrix,
    position_measurement_noise,
    initial_state_covariance,
)


def test_transition_matrix():

    dt = 0.5

    F = constant_velocity_transition_matrix(dt)

    assert F.shape == (6, 6)

    assert F[0, 3] == dt
    assert F[1, 4] == dt
    assert F[2, 5] == dt


def test_process_noise():

    Q = constant_velocity_process_noise(
        dt=0.1,
        acceleration_variance=2.0,
    )

    assert Q.shape == (6, 6)

    assert np.all(np.isfinite(Q))

    assert np.allclose(Q, Q.T)


def test_measurement_matrix():

    H = position_measurement_matrix()

    assert H.shape == (3, 6)

    assert np.array_equal(
        H,
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        ),
    )


def test_measurement_noise():

    R = position_measurement_noise(4.0)

    assert R.shape == (3, 3)

    assert np.allclose(R, R.T)

    assert np.all(np.diag(R) == 4.0)


def test_initial_covariance():

    P = initial_state_covariance(
        position_variance=25.0,
        velocity_variance=100.0,
    )

    assert P.shape == (6, 6)

    assert P[0, 0] == 25.0
    assert P[1, 1] == 25.0
    assert P[2, 2] == 25.0

    assert P[3, 3] == 100.0
    assert P[4, 4] == 100.0
    assert P[5, 5] == 100.0