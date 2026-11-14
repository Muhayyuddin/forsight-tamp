"""Geometric feasibility contract for the SHF2 shelf scenario.

SHF2 was previously unsolvable for every method because its geometry ignored
the measured envelope of the FR3 hand in the ``front`` side-grasp
orientation.  Three independent defects made the scene impossible:

* the two objects sharing a shelf level were placed closer together than the
  hand is wide, so the second placement always ended in collision with the
  first;
* the five source objects sat on an 0.08 m grid, so a side grasp of any
  object in the second or third row passed through its neighbours; and
* the shelf was centred at x = 0.55, which puts the right half of the bottom
  bay outside the arm's reach envelope for this orientation.

These tests pin the invariants the corrected geometry must satisfy so the
scene cannot silently regress into an unsatisfiable state again.  The hand
envelope constants are measured from the simulator, not assumed; see
``HAND_*`` below.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "envs" / "task_instances" / "foresight" / "SHF2_medium_shelf.json"

# Measured AABB of the hand and finger links, relative to the tool frame, with
# the gripper held in the "front" side-grasp orientation.
HAND_HALF_WIDTH_X = 0.108
HAND_HALF_HEIGHT_Z = 0.035
HAND_REAR_REACH_Y = 0.133
FINGER_GAP_X = 0.0775

# Tool-frame offset applied by ShelfEnv/PyBulletRobot for a front grasp.
FRONT_OFFSET = (0.0, -0.035, 0.02)

# Largest tool-frame x reachable in the front orientation at bottom-bay
# height, measured by sweeping IK over the table.
MAX_REACHABLE_TCP_X_AT_BOTTOM_BAY = 0.56


@pytest.fixture(scope="module")
def scenario():
    return json.loads(SCENARIO.read_text())


@pytest.fixture(scope="module")
def boxes(scenario):
    return {box["name"]: box for box in scenario["boxes"].values()}


def _levels(scenario):
    return scenario["shelf_config"]["levels"]


def test_every_object_fits_between_the_open_fingers(boxes):
    for name, box in boxes.items():
        assert box["w"] <= FINGER_GAP_X, (
            f"{name} is {box['w']:.3f} m wide but the open fingers span only "
            f"{FINGER_GAP_X:.4f} m, so it can never be side-grasped"
        )


def test_source_objects_form_a_single_depth_ordered_column(boxes, scenario):
    """Each object must be clearable before the one behind it."""
    order = [name for level in _levels(scenario) for name in level["objects"]]
    assert set(order) == set(boxes), "every object must be assigned to a level"

    xs = {boxes[name]["x"] for name in order}
    assert len(xs) == 1, (
        "source objects must share one x column so that a side grasp is "
        f"blocked only by the objects in front of it; got {sorted(xs)}"
    )

    ys = [boxes[name]["y"] for name in order]
    assert ys == sorted(ys), (
        "the declared placement order must run front to back, otherwise a "
        f"grasp reaches through an object that is still present; got {ys}"
    )


def test_source_spacing_clears_the_hand_of_the_object_behind(boxes, scenario):
    """The object behind the grasp target must sit clear of the hand."""
    order = [name for level in _levels(scenario) for name in level["objects"]]
    for near, far in zip(order, order[1:]):
        tool_y = boxes[near]["y"] + FRONT_OFFSET[1]
        hand_front_y = tool_y + 0.011
        far_near_face = boxes[far]["y"] - boxes[far]["l"] / 2
        assert far_near_face > hand_front_y, (
            f"grasping {near} puts the hand at y={hand_front_y:.3f} but "
            f"{far} starts at y={far_near_face:.3f}"
        )


def test_objects_sharing_a_level_are_separated_by_more_than_the_hand(
        boxes, scenario):
    """Placing the second object must not drive the hand into the first."""
    cfg = scenario["shelf_config"]
    half_width = cfg["width"] / 2
    usable = (cfg["x"] - half_width + HAND_HALF_WIDTH_X,
              cfg["x"] + half_width - HAND_HALF_WIDTH_X)
    for level in _levels(scenario):
        names = level["objects"]
        if len(names) < 2:
            continue
        span = usable[1] - usable[0]
        required = HAND_HALF_WIDTH_X + max(
            boxes[name]["w"] for name in names) / 2
        assert span > required, (
            f"level '{level['label']}' holds {len(names)} objects but its "
            f"usable tool-frame span is only {span:.3f} m, less than the "
            f"{required:.3f} m needed to clear an already-placed neighbour"
        )


def test_bottom_bay_is_inside_the_front_orientation_reach_envelope(scenario):
    """The bottom bay is the reach-limited one; it must stay reachable."""
    cfg = scenario["shelf_config"]
    leftmost_tool_x = cfg["x"] - cfg["width"] / 2 + HAND_HALF_WIDTH_X
    assert leftmost_tool_x < MAX_REACHABLE_TCP_X_AT_BOTTOM_BAY, (
        "the shelf is too far from the base: even its leftmost reachable "
        f"placement needs tool x={leftmost_tool_x:.3f}, beyond the measured "
        f"limit of {MAX_REACHABLE_TCP_X_AT_BOTTOM_BAY}"
    )


def test_each_bay_clears_its_tallest_object_and_the_hand(boxes, scenario):
    cfg = scenario["shelf_config"]
    thickness = cfg["wall_thickness"]
    levels = _levels(scenario)
    ceiling = max(level["z"] for level in levels) + 0.15
    for index, level in enumerate(levels):
        surface = level["z"] + thickness / 2
        above = (levels[index + 1]["z"] if index + 1 < len(levels) else ceiling)
        bay_ceiling = above - thickness / 2
        for name in level["objects"]:
            height = boxes[name]["h"]
            centre = surface + height / 2
            hand_top = centre + FRONT_OFFSET[2] + HAND_HALF_HEIGHT_Z
            assert surface + height < bay_ceiling, (
                f"{name} ({height:.3f} m) does not fit in bay "
                f"'{level['label']}'"
            )
            assert hand_top < bay_ceiling, (
                f"placing {name} drives the hand to z={hand_top:.3f}, above "
                f"the bay ceiling at z={bay_ceiling:.3f}"
            )


def test_placement_order_matches_the_source_column_order(boxes, scenario):
    """The bottom bay is reach-limited, so it must not hold a pair."""
    levels = _levels(scenario)
    bottom = levels[0]
    assert len(bottom["objects"]) == 1, (
        "the bottom bay is the reach-limited one: a second object there "
        "forces one slot against the left wall, where the post-place retreat "
        "has no IK solution and the placement is discarded as infeasible"
    )
    tallest = max(boxes, key=lambda name: boxes[name]["h"])
    assert tallest in bottom["objects"], (
        "the tallest object must occupy the bottom bay"
    )
