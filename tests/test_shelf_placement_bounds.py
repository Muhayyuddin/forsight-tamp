"""The planner-facing shelf bounds must exclude the strip the hand cannot use.

ACFM's hard masks test the object footprint against bounds, collision,
reachability and support.  None of them models the gripper, so a candidate
flush against a shelf side wall passes every mask while leaving the hand
inside the wall.  Under greedy top-ranked selection the entire rollout quota
was consumed by one such wall-adjacent block, and the step reported no
hard-feasible candidate even though feasible placements existed further in.
"""
import pytest

from envs.shelf_env import (
    GRIPPER_DEPTH_CLEARANCE,
    GRIPPER_SIDE_CLEARANCE,
    ShelfEnv,
)


def _env(width=0.5, depth=0.18):
    env = ShelfEnv.__new__(ShelfEnv)
    env._shelf_config = {
        "x": 0.5,
        "y": 0.2,
        "width": width,
        "depth": depth,
        "wall_thickness": 0.01,
        "levels": [{"z": 0.04, "label": "bottom", "objects": []}],
    }
    return env


def test_planner_bounds_are_inset_by_the_gripper_clearance():
    bounds = _env().get_shelf_bounds()
    assert bounds["x_min"] == pytest.approx(0.25 + GRIPPER_SIDE_CLEARANCE)
    assert bounds["x_max"] == pytest.approx(0.75 - GRIPPER_SIDE_CLEARANCE)
    assert bounds["y_min"] == pytest.approx(0.11 + GRIPPER_DEPTH_CLEARANCE)
    assert bounds["y_max"] == pytest.approx(0.29 - GRIPPER_DEPTH_CLEARANCE)


def test_raw_extent_is_still_available_for_structural_queries():
    bounds = _env().get_shelf_bounds(gripper_clearance=False)
    assert bounds["x_min"] == pytest.approx(0.25)
    assert bounds["x_max"] == pytest.approx(0.75)
    assert bounds["y_min"] == pytest.approx(0.11)
    assert bounds["y_max"] == pytest.approx(0.29)


def test_level_surface_height_is_unchanged_by_the_inset():
    env = _env()
    assert env.get_shelf_bounds(level_idx=0)["z"] == pytest.approx(
        env.get_shelf_bounds(level_idx=0, gripper_clearance=False)["z"])


def test_a_shelf_too_narrow_for_the_hand_is_rejected_not_silently_empty():
    """An empty candidate region must not masquerade as a planning failure."""
    with pytest.raises(ValueError, match="too small to admit the hand"):
        _env(width=2 * GRIPPER_SIDE_CLEARANCE).get_shelf_bounds()
    with pytest.raises(ValueError, match="too small to admit the hand"):
        _env(depth=2 * GRIPPER_DEPTH_CLEARANCE).get_shelf_bounds()
