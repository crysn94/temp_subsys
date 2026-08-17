import numpy as np

from tracking.models.state_ca import StateCA


def test_state_ca():

    state = StateCA(
        state=np.array([
            100.0,
            200.0,
            300.0,
            10.0,
            20.0,
            30.0,
            1.0,
            2.0,
            3.0,
        ]),
        covariance=np.eye(9),
    )

    assert state.x == 100.0
    assert state.y == 200.0
    assert state.z == 300.0

    assert state.vx == 10.0
    assert state.vy == 20.0
    assert state.vz == 30.0

    assert state.ax == 1.0
    assert state.ay == 2.0
    assert state.az == 3.0

    assert state.state_dimension == 9
    assert state.model_name == "constant_acceleration_3d"