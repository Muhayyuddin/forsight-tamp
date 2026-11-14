import numpy as np
import pytest

import utils.planning_util as planning_util


def test_densify_joint_path_bounds_every_joint_step(monkeypatch):
    monkeypatch.setattr(
        planning_util,
        "get_difference_fn",
        lambda _body, _joints: (
            lambda q2, q1: np.asarray(q2) - np.asarray(q1)
        ),
    )
    sparse = [
        (0.0, 0.0, 0.0),
        (1.25, -0.4, 0.1),
        (1.30, -0.2, 0.1),
    ]

    dense = planning_util.densify_joint_path(
        body=1,
        joints=[0, 1, 2],
        path=sparse,
        max_joint_step=0.025,
    )

    steps = np.abs(np.diff(np.asarray(dense), axis=0))
    assert np.max(steps) <= 0.025 + 1e-12
    assert np.allclose(dense[0], sparse[0])
    assert np.allclose(dense[-1], sparse[-1])


def test_densify_joint_path_rejects_nonfinite_values(monkeypatch):
    monkeypatch.setattr(
        planning_util,
        "get_difference_fn",
        lambda _body, _joints: (
            lambda q2, q1: np.asarray(q2) - np.asarray(q1)
        ),
    )

    with pytest.raises(ValueError, match="non-finite"):
        planning_util.densify_joint_path(
            body=1,
            joints=[0, 1],
            path=[(0.0, 0.0), (np.nan, 0.2)],
        )


def test_time_parameterization_limits_velocity_and_acceleration():
    time_step = 1.0 / 60.0
    path = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, -0.4),
    ]

    timed = planning_util.time_parameterize_joint_path(
        path,
        time_step=time_step,
        max_joint_velocity=0.5,
        max_joint_acceleration=2.0,
    )
    timed = np.asarray(timed)
    velocity = np.diff(timed, axis=0) / time_step
    acceleration = np.diff(velocity, axis=0) / time_step

    assert np.allclose(timed[0], path[0])
    assert np.allclose(timed[-1], path[-1])
    assert np.max(np.abs(velocity)) <= 0.5 + 1e-3
    assert np.max(np.abs(acceleration)) <= 2.0 + 1e-2

    # The exact corner is retained and approached/departed at nearly zero
    # speed, rather than reversing direction in one playback frame.
    corner_index = np.flatnonzero(
        np.all(np.isclose(timed, path[1], atol=1e-12), axis=1)
    )[0]
    assert np.linalg.norm(velocity[corner_index - 1]) < 0.01
    assert np.linalg.norm(velocity[corner_index]) < 0.01
