from types import SimpleNamespace

import main_foresight
from main_foresight import ForeSightTAMP
from utils.tamp_util import (
    Parameter,
    PrimitiveAction,
    TAMPFeedback,
    text_to_actions,
)


def _planner_for_slot_generation(monkeypatch, robot_xy):
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(
        robot=SimpleNamespace(robot=7),
        _basket_config={},
    )
    planner._packing_slot_strategy = None
    planner._declared_xy_size = lambda _name: (0.07, 0.07, True)
    monkeypatch.setattr(
        main_foresight.p,
        "getBasePositionAndOrientation",
        lambda _robot: ((*robot_xy, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    return planner


def _primitive_actions():
    return {
        "pick": PrimitiveAction(name="pick", obj_arity=1),
        "place": PrimitiveAction(
            name="place",
            obj_arity=1,
            parameters={
                "x": Parameter("x", 0.0, 1.4),
                "y": Parameter("y", -1.0, 1.0),
                "theta": Parameter("theta", -3.14, 3.14),
            },
        ),
    }


def test_dense_slots_are_ordered_from_the_robot_side(monkeypatch):
    planner = _planner_for_slot_generation(monkeypatch, robot_xy=(0.0, 0.0))

    strategy = planner._compute_packing_slot_strategy(
        [f"can_{i}" for i in range(1, 7)],
        {"x_min": 0.29, "x_max": 0.69, "y_min": -0.13, "y_max": 0.13},
    )

    slot_x = [slot["x"] for slot in strategy["slots"]]
    assert slot_x == sorted(slot_x)
    assert strategy["access_origin"] == {"x": 0.0, "y": 0.0}


def test_dense_slot_order_tracks_a_robot_on_the_opposite_side(monkeypatch):
    planner = _planner_for_slot_generation(monkeypatch, robot_xy=(1.0, 0.0))

    strategy = planner._compute_packing_slot_strategy(
        [f"can_{i}" for i in range(1, 7)],
        {"x_min": 0.29, "x_max": 0.69, "y_min": -0.13, "y_max": 0.13},
    )

    slot_x = [slot["x"] for slot in strategy["slots"]]
    assert slot_x == sorted(slot_x, reverse=True)


def _mixed_planner(monkeypatch, basket, sizes, grasp_theta=None):
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(
        robot=SimpleNamespace(robot=7),
        _basket_config=basket,
        _grasp_theta_dict=grasp_theta or {},
    )
    planner._packing_slot_strategy = None
    planner._declared_xy_size = lambda name: (*sizes[name], False)
    monkeypatch.setattr(
        main_foresight.p,
        "getBasePositionAndOrientation",
        lambda _robot: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    return planner


def _assert_mixed_slots_are_valid(strategy):
    bounds = strategy["usable_bounds"]
    gap = strategy["object_clearance"]
    rectangles = []
    for slot in strategy["slots"]:
        half_x = slot["footprint_x"] / 2.0
        half_y = slot["footprint_y"] / 2.0
        rect = {
            "x_min": slot["x"] - half_x,
            "x_max": slot["x"] + half_x,
            "y_min": slot["y"] - half_y,
            "y_max": slot["y"] + half_y,
        }
        assert rect["x_min"] >= bounds["x_min"] - 1e-4
        assert rect["x_max"] <= bounds["x_max"] + 1e-4
        assert rect["y_min"] >= bounds["y_min"] - 1e-4
        assert rect["y_max"] <= bounds["y_max"] + 1e-4
        for other in rectangles:
            assert (
                rect["x_max"] + gap <= other["x_min"] + 1e-4
                or other["x_max"] + gap <= rect["x_min"] + 1e-4
                or rect["y_max"] + gap <= other["y_min"] + 1e-4
                or other["y_max"] + gap <= rect["y_min"] + 1e-4
            )
        rectangles.append(rect)


def test_a2_mixed_objects_receive_one_global_nonoverlapping_layout(monkeypatch):
    sizes = {
        "cracker_box": (0.06, 0.16),
        "sugar_box": (0.04, 0.09),
        "tomato_soup_can": (0.07, 0.07),
        "gelatin_box": (0.0723, 0.09),
    }
    planner = _mixed_planner(
        monkeypatch,
        basket={"x": 0.54, "y": 0.0, "w": 0.288, "l": 0.21},
        sizes=sizes,
        grasp_theta={"cracker_box": 1.5708},
    )

    strategy = planner._compute_packing_slot_strategy(
        list(sizes),
        {"x_min": 0.396, "x_max": 0.684, "y_min": -0.105, "y_max": 0.105},
    )

    assert strategy["layout_type"] == "mixed_rectangles"
    assert len(strategy["slots"]) == len(sizes)
    assert strategy["slots"][0]["object"] == "cracker_box"
    _assert_mixed_slots_are_valid(strategy)


def test_pkg3_mixed_layout_retains_rectangular_potted_can(monkeypatch):
    sizes = {
        "cracker_box": (0.06, 0.16),
        "sugar_box": (0.04, 0.09),
        "tomato_soup_can": (0.07, 0.07),
        "potted_meat_can": (0.0753, 0.10),
        "tuna_fish_can": (0.09, 0.09),
        "gelatin_box": (0.0681, 0.09),
    }
    planner = _mixed_planner(
        monkeypatch,
        basket={"x": 0.48, "y": 0.0, "w": 0.42, "l": 0.28},
        sizes=sizes,
        grasp_theta={"cracker_box": 1.5708},
    )

    strategy = planner._compute_packing_slot_strategy(
        list(sizes),
        {"x_min": 0.27, "x_max": 0.69, "y_min": -0.14, "y_max": 0.14},
    )

    assert strategy["layout_type"] == "mixed_rectangles"
    assert len(strategy["slots"]) == len(sizes)
    potted = next(
        slot for slot in strategy["slots"]
        if slot["object"] == "potted_meat_can"
    )
    assert sorted((potted["footprint_x"], potted["footprint_y"])) == [
        0.0753, 0.1
    ]
    potted_rotations = {
        slot["theta"]
        for layout in strategy["candidate_slots"]
        for slot in layout
        if slot["object"] == "potted_meat_can"
    }
    assert potted_rotations == {0.0, 1.5708}
    _assert_mixed_slots_are_valid(strategy)


def test_invalid_llm_dense_plan_is_replaced_with_computed_slots():
    primitives = _primitive_actions()
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(primitive_actions=primitives)
    planner._task_type = "packing"
    planner._packing_slot_strategy = {
        "slots": [
            {
                "object": "can_1",
                "x": 0.35,
                "y": -0.06,
                "theta": 0.0,
            },
            {
                "object": "can_2",
                "x": 0.35,
                "y": 0.06,
                "theta": 0.0,
            },
        ],
    }
    invalid = text_to_actions(
        [
            "pick(['can_1'], {})",
            "place(['can_1'], {'x': 0.40, 'y': 0.0, 'theta': 0.0})",
            "pick(['can_2'], {})",
            "place(['can_2'], {'x': 0.41, 'y': 0.0, 'theta': 0.0})",
        ],
        primitives,
    )

    repaired = planner._repair_packing_slot_plan(invalid)

    assert repaired is not invalid
    assert [action.primitive.name for action in repaired] == [
        "pick", "place", "pick", "place"
    ]
    assert [
        (action.obj_args[0], action.param_args["x"], action.param_args["y"])
        for action in repaired
        if action.primitive.name == "place"
    ] == [
        ("can_1", 0.35, -0.06),
        ("can_2", 0.35, 0.06),
    ]


def test_matching_llm_dense_plan_is_preserved():
    primitives = _primitive_actions()
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(primitive_actions=primitives)
    planner._task_type = "packing"
    slots = [
        {"object": "can_1", "x": 0.35, "y": -0.06, "theta": 0.0},
        {"object": "can_2", "x": 0.35, "y": 0.06, "theta": 0.0},
    ]
    planner._packing_slot_strategy = {"slots": slots}
    valid = text_to_actions(
        [
            "pick(['can_1'], {})",
            "place(['can_1'], {'x': 0.35, 'y': -0.06, 'theta': 0.0})",
            "pick(['can_2'], {})",
            "place(['can_2'], {'x': 0.35, 'y': 0.06, 'theta': 0.0})",
        ],
        primitives,
    )

    assert planner._repair_packing_slot_plan(valid) is valid


def test_matching_coordinates_with_wrong_rotation_are_repaired():
    primitives = _primitive_actions()
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(primitive_actions=primitives)
    planner._task_type = "packing"
    slots = [{
        "object": "cracker_box",
        "x": 0.45,
        "y": 0.0,
        "theta": 1.5708,
    }]
    planner._packing_slot_strategy = {
        "layout_type": "mixed_rectangles",
        "slots": slots,
    }
    wrong_rotation = text_to_actions(
        [
            "pick(['cracker_box'], {})",
            (
                "place(['cracker_box'], "
                "{'x': 0.45, 'y': 0.0, 'theta': 0.0})"
            ),
        ],
        primitives,
    )

    repaired = planner._repair_packing_slot_plan(wrong_rotation)

    place = repaired[1]
    assert place.param_args["theta"] == 1.5708


def test_pick_failure_triggers_rehearsal_validated_object_reordering(monkeypatch):
    primitives = _primitive_actions()
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace(primitive_actions=primitives)
    planner._task_type = "packing"
    slots = [
        {"object": "can_1", "x": 0.35, "y": -0.06, "theta": 0.0},
        {"object": "can_2", "x": 0.35, "y": 0.06, "theta": 0.0},
        {"object": "can_3", "x": 0.45, "y": -0.06, "theta": 0.0},
    ]
    planner._packing_slot_strategy = {
        "slots": slots,
        "access_origin": {"x": 0.0, "y": 0.0},
    }
    original = planner._build_packing_slot_plan(slots)
    failed_pick = original[2]
    feedback = TAMPFeedback(
        motion_planner_feedback="Failed because no collision-free trajectory is found",
        action_success=False,
    )

    rehearsed_orders = []

    class FakeRehearsal:
        def __init__(self, _env):
            pass

        def rehearse(self, candidate):
            order = [
                action.obj_args[0]
                for action in candidate
                if action.primitive.name == "pick"
            ]
            rehearsed_orders.append(order)
            return SimpleNamespace(success=order[0] == "can_2")

    monkeypatch.setattr(main_foresight, "MentalRehearsal", FakeRehearsal)

    repaired = planner._repair_packing_slot_plan(
        original, [(failed_pick, feedback)])

    repaired_order = [
        action.obj_args[0]
        for action in repaired
        if action.primitive.name == "pick"
    ]
    assert rehearsed_orders[0] == ["can_2", "can_1", "can_3"]
    assert repaired_order == ["can_2", "can_1", "can_3"]
    assert [
        slot["object"] for slot in planner._packing_slot_strategy["slots"]
    ] == repaired_order
