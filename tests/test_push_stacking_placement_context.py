"""Regression tests for push-stacking placement context construction.

The corrected sequential planner calls ``_placement_acfm_context`` for every
parameterised action.  ``PushStackingEnv`` is neither a ``ShelfEnv`` nor a
``StackingEnv``, so before a dedicated branch existed it fell through to the
packing case and queried a ``basket`` body that push scenes never create.  That
raised ``KeyError: 'basket'``, which ``generate_results.run_trial`` recorded as
an ordinary unsuccessful trial -- making a crash indistinguishable from a
planning failure on every PUSH scenario.
"""

import pytest

from envs.push_stacking_env import PushStackingEnv
from main_foresight import ForeSightTAMP


BASE_MIN, BASE_MAX = (0.50, -0.06, 0.00), (0.62, 0.06, 0.05)


def _env(stacking_order=("large_base", "small_cube"), tolerance=None):
    env = PushStackingEnv.__new__(PushStackingEnv)
    env._objects = {
        "large_base": 11,
        "small_cube": 12,
        "table": 13,
        "target_marker": 14,
    }
    env._stacking_order = list(stacking_order)
    env._target_pose = {"x": 0.56, "y": 0.0}
    if tolerance is not None:
        env._target_pose["stack_xy_tolerance"] = tolerance
    env.get_bb = lambda name: (BASE_MIN, BASE_MAX)
    env.is_fixture = lambda name: name == "target_marker"
    return env


def _planner(env):
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = env
    planner._packing_slot_strategy = None
    # Bottom of a 0.04 m tall cube resting on the base top face.
    planner._body_z_with_bottom_on = lambda name, support_z: support_z + 0.02
    return planner


def test_push_stacking_placement_context_does_not_query_a_basket():
    context = _planner(_env())._placement_acfm_context("small_cube")

    base_cx = (BASE_MIN[0] + BASE_MAX[0]) / 2.0
    base_cy = (BASE_MIN[1] + BASE_MAX[1]) / 2.0
    bounds = context["bounds"]
    assert bounds["x_min"] == pytest.approx(base_cx - 0.04)
    assert bounds["x_max"] == pytest.approx(base_cx + 0.04)
    assert bounds["y_min"] == pytest.approx(base_cy - 0.04)
    assert bounds["y_max"] == pytest.approx(base_cy + 0.04)
    assert bounds["z_floor"] == pytest.approx(BASE_MAX[2])


def test_placement_region_follows_the_env_stack_tolerance():
    context = _planner(_env(tolerance=0.01))._placement_acfm_context("small_cube")
    bounds = context["bounds"]
    assert bounds["x_max"] - bounds["x_min"] == pytest.approx(0.02)
    assert bounds["y_max"] - bounds["y_min"] == pytest.approx(0.02)


def test_placement_heights_include_release_clearance_above_the_base():
    """At the exact resting height the gripper collides with the base.

    Pinning z to the contact height made every candidate fail the simulator
    rollout with "end configuration is in collision with object large_base".
    The archived successful push-stack plans release 0.010 m above the base
    top, so that clearance must be among the sampled heights.
    """
    context = _planner(_env())._placement_acfm_context("small_cube")

    nominal = BASE_MAX[2] + 0.02
    assert context["z_values"][0] == pytest.approx(nominal)
    assert any(z == pytest.approx(nominal + 0.010) for z in context["z_values"])
    # Strictly increasing, and never below the support surface.
    assert context["z_values"] == sorted(context["z_values"])
    assert min(context["z_values"]) >= BASE_MAX[2]


def test_support_is_the_base_top_face_and_ground_support_is_refused():
    context = _planner(_env())._placement_acfm_context("small_cube")

    (surface,) = context["support_surfaces"]
    assert surface["z"] == pytest.approx(BASE_MAX[2])
    assert surface["label"] == "top face of large_base"
    assert context["allow_ground_support"] is False
    assert context["require_configured_support"] is True
    assert context["is_3d"] is True
    assert context["z_values"][0] == pytest.approx(BASE_MAX[2] + 0.02)


def test_base_and_fixtures_are_excluded_from_collision_context():
    context = _planner(_env())._placement_acfm_context("small_cube")
    # The base is the support, not an obstacle; the marker is a visual fixture.
    assert set(context["existing"]) == set()


def test_missing_base_object_fails_loudly_rather_than_silently():
    with pytest.raises(RuntimeError, match="requires a base object"):
        _planner(_env(stacking_order=()))._placement_acfm_context("small_cube")


def test_already_at_target_sentinel_is_shared_between_env_and_planner():
    """The planner compares feedback text against the env's sentinel.

    A push whose target is already satisfied succeeds without moving and so
    emits no push_contact phase. The planner accepts it only by matching this
    exact string, so the two modules must not drift apart.
    """
    import inspect

    from envs.push_stacking_env import PUSH_ALREADY_AT_TARGET
    import main_foresight

    assert PUSH_ALREADY_AT_TARGET == "Already at push target."
    # The planner imports the sentinel rather than hard-coding the literal.
    source = inspect.getsource(main_foresight.ForeSightTAMP)
    assert "PUSH_ALREADY_AT_TARGET" in source
    assert '"Already at push target."' not in source
