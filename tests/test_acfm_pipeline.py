import json
from types import SimpleNamespace

import numpy as np
import pytest

import acfm as acfm_module
import main_foresight
from acfm import ACFM
from main_foresight import (
    CandidateSelectionError,
    ForeSightTAMP,
    _parse_action_skeleton_response,
    _parse_candidate_id_response,
)
from utils.tamp_util import Parameter, PrimitiveAction, TAMPFeedback


def _primitives():
    return {
        "pick": PrimitiveAction("pick", obj_arity=1),
        "place": PrimitiveAction(
            "place",
            obj_arity=1,
            parameters={
                "x": Parameter("x", -1.0, 1.0),
                "y": Parameter("y", -1.0, 1.0),
                "theta": Parameter("theta", -3.14, 3.14),
            },
        ),
    }


def test_hard_masks_are_applied_before_soft_ranking(monkeypatch):
    mapper = ACFM(resolution=2)
    monkeypatch.setattr(
        acfm_module.p,
        "getAABB",
        lambda _body: ((-0.05, -0.05, 0.0), (0.05, 0.05, 0.1)),
    )
    monkeypatch.setattr(
        acfm_module.p,
        "getBasePositionAndOrientation",
        lambda _body: ((0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(
        acfm_module.p, "getEulerFromQuaternion", lambda _orn: (0.0, 0.0, 0.0))
    monkeypatch.setattr(
        mapper,
        "_bounds_mask_2d",
        lambda: np.array([[True, True], [True, False]], dtype=bool),
    )
    monkeypatch.setattr(
        mapper,
        "_collision_layer",
        lambda *_args: np.array([[0.0, 1.0], [0.0, 0.0]]),
    )
    monkeypatch.setattr(
        mapper,
        "_reachability_layer",
        lambda *_args, **_kwargs: np.array([[1.0, 1.0], [0.0, 1.0]]),
    )
    monkeypatch.setattr(
        mapper,
        "_stability_layer",
        lambda: np.ones((2, 2)),
    )

    score = mapper.compute(
        target_obj_id=1,
        target_name="rect",
        existing_objects={},
        robot_id=2,
        ee_link_idx=3,
        basket_bounds={
            "x_min": -0.5,
            "x_max": 0.5,
            "y_min": -0.5,
            "y_max": 0.5,
        },
    )

    assert set(mapper.hard_masks) == {
        "bounds", "collision", "reachability", "support", "valid_contact"
    }
    assert mapper.feasible_mask.tolist() == [[True, False], [False, False]]
    assert score[0, 0] > 0.0
    assert np.count_nonzero(score) == 1
    for layer in mapper.soft_layers.values():
        assert np.all(layer[~mapper.feasible_mask] == 0.0)


def test_rectangular_parameter_domain_includes_yaw_and_symbolic_ids():
    mapper = ACFM(resolution=2)
    mapper.target_name = "rect"
    mapper.target_dims = [0.20, 0.08, 0.05]
    mapper.target_aabb_offset = [0.0, 0.0]
    mapper.target_reference_yaw = 0.0
    mapper.bounds = {
        "x_min": -0.5,
        "x_max": 0.5,
        "y_min": -0.5,
        "y_max": 0.5,
    }
    mapper.x_range = np.array([-0.2, 0.2])
    mapper.y_range = np.array([-0.2, 0.2])
    mapper.grid = np.array([[1.0, 0.9], [0.8, 0.7]])
    mapper.feasible_mask = np.ones((2, 2), dtype=bool)
    mapper.hard_masks = {"bounds": mapper.feasible_mask.copy()}
    mapper.contact_modes = ("top",)
    mapper.existing_footprints = []

    rows = mapper.get_parameter_candidates(k=4)

    assert [row["candidate_id"] for row in rows] == [
        "C0001", "C0002", "C0003", "C0004"
    ]
    assert {row["theta"] for row in rows} == {0.0, 1.5708}
    assert {row["contact_mode"] for row in rows} == {"top"}


def test_oriented_3d_check_allows_xy_overlap_with_support_below():
    mapper = ACFM(resolution=2)
    mapper.target_dims = [0.10, 0.08, 0.04]
    mapper.target_aabb_offset = [0.0, 0.0]
    mapper.target_reference_yaw = 0.0
    mapper.bounds = {
        "x_min": -0.2,
        "x_max": 0.2,
        "y_min": -0.2,
        "y_max": 0.2,
    }
    mapper.existing_footprints = [{
        "cx": 0.0,
        "cy": 0.0,
        "hx": 0.06,
        "hy": 0.05,
        "geometry_yaw": 0.0,
        "z_min": 0.0,
        "z_max": 0.04,
    }]

    assert not mapper._oriented_candidate_is_endpoint_feasible(
        0.0, 0.0, 0.0)
    assert mapper._oriented_candidate_is_endpoint_feasible(
        0.0, 0.0, 0.0, z=0.061)


def test_configured_support_uses_existing_coverage_tiers():
    mapper = acfm_module.ACFM3D(resolution=1, z_resolution=1)
    mapper.x_range = np.array([0.0])
    mapper.y_range = np.array([0.0])
    mapper.z_range = np.array([0.08])
    mapper.target_dims = [0.06, 0.10, 0.08]
    mapper.target_aabb_offset = [0.0, 0.0]
    mapper.target_aabb_z_offset = 0.0
    mapper.bounds = {
        "x_min": -0.2,
        "x_max": 0.2,
        "y_min": -0.2,
        "y_max": 0.2,
    }
    mapper.allow_ground_support = False
    mapper.require_configured_support = True
    mapper.support_surfaces = [{
        "x_min": -0.045,
        "x_max": 0.045,
        "y_min": -0.0425,
        "y_max": 0.0425,
        "z": 0.04,
    }]

    support = mapper._support_layer(target_obj_id=1, existing_objects={})

    # The target overhangs by 7.5 mm on each Y side but retains 85% support.
    assert support[0, 0, 0] == pytest.approx(1.0)
    assert mapper.support_score_at(0.0, 0.0, 0.08) == pytest.approx(1.0)


def test_candidate_response_accepts_only_an_offered_symbolic_id():
    allowed = ["A02C001", "A02C002"]
    raw = json.dumps({"Reasoning": "best margin", "Candidate ID": "A02C002"})
    assert _parse_candidate_id_response(raw, allowed) == "A02C002"

    with pytest.raises(CandidateSelectionError, match="not in the offered set"):
        _parse_candidate_id_response(
            json.dumps({"Reasoning": "invented", "Candidate ID": "A02C999"}),
            allowed,
        )
    with pytest.raises(CandidateSelectionError, match="exactly"):
        _parse_candidate_id_response(
            json.dumps({
                "Reasoning": "copied coordinates",
                "Candidate ID": "A02C001",
                "x": 0.3,
            }),
            allowed,
        )


def test_llm_candidate_rows_hide_controller_coordinates():
    rows = ForeSightTAMP._candidate_prompt_rows([{
        "candidate_id": "A02C001",
        "object": "box",
        "x": 0.31,
        "y": -0.04,
        "z": 0.12,
        "theta": 1.57,
        "grasp_mode": "front",
        "contact_mode": "front",
        "score": 0.86,
        "hard_constraints": {"reachability": True},
    }])

    assert rows == [{
        "candidate_id": "A02C001",
        "object": "box",
        "grasp_mode": "front",
        "contact_mode": "front",
        "score": 0.86,
        "hard_constraints": {"reachability": True},
    }]
    assert not {"x", "y", "z", "theta"} & set(rows[0])


def test_candidate_audit_rows_include_heldout_sweep_metadata():
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner._audit_metadata = {
        "scenario_id": "A1__heldout_001",
        "split": "held_out",
        "domain": "packing",
        "resolution": 20,
        "weight_profile": "2d_default",
        "threshold": None,
        "push_cost_coefficient": None,
    }
    rows = planner._candidate_audit_rows([{
        "candidate_id": "A02C001",
        "object": "box",
        "x": 0.3,
        "y": 0.0,
        "theta": 0.0,
        "score": 0.8,
        "hard_constraints": {"bounds": True},
        "endpoint_map_feasible": True,
        "ground_truth_feasible": False,
    }])

    assert rows[0]["scenario_id"] == "A1__heldout_001"
    assert rows[0]["split"] == "held_out"
    assert rows[0]["resolution"] == 20
    assert rows[0]["ground_truth_feasible"] is False


def test_audit_metadata_accepts_generator_benchmark_split_alias():
    metadata = ForeSightTAMP._build_audit_metadata(
        {
            "scenario_id": "A1__heldout_001",
            "split": None,
            "benchmark_split": "held_out",
        },
        task_type="packing",
        resolution=12,
    )

    assert metadata["split"] == "held_out"
    assert metadata["resolution"] == 12
    assert metadata["weight_profile"] == "2d_default"


def test_selected_candidate_score_uses_the_decision_row():
    history = [{
        "step": 3,
        "action": "place",
        "object": "box",
        "candidate_rows": [
            {"candidate_id": "A04C001", "score": 0.92},
            {"candidate_id": "A04C002", "score": 0.41},
        ],
        "decision": {"candidate_id": "A04C002"},
    }]

    assert ForeSightTAMP._selected_candidate_score(
        history, step=3, primitive="place", obj_name="box") == 0.41


def test_action_skeleton_rejects_spatial_parameters():
    primitives = _primitives()
    valid = json.dumps({
        "Reasoning": "pick then place",
        "Action Skeleton": ["pick(['box'], {})", "place(['box'], {})"],
    })
    actions, _ = _parse_action_skeleton_response(valid, primitives)
    assert [action.primitive.name for action in actions] == ["pick", "place"]

    invalid = json.dumps({
        "Reasoning": "contains a coordinate",
        "Action Skeleton": [
            "pick(['box'], {})",
            "place(['box'], {'x': 0.2})",
        ],
    })
    with pytest.raises(ValueError, match="must not contain coordinates"):
        _parse_action_skeleton_response(invalid, primitives)


def test_sequential_top1_advances_predicted_state_and_records_every_decision(
        monkeypatch):
    primitives = _primitives()

    class FakeEnv:
        def __init__(self):
            self.primitive_actions = primitives
            self.objects = {"box": 1}
            self.theta_dict = {}
            self.robot = SimpleNamespace(
                attachments_robot=[],
                last_grasp_direction=None,
                last_grasp_position_offset=None,
                last_motion_trace=[],
            )
            self.state = []

        def get_observation(self):
            return {"events": list(self.state)}

        def step(self, action, play_traj=False):
            self.state.append(action.primitive.name)
            if action.primitive.name == "place":
                self.robot.last_motion_trace = [
                    {"phase": "place_transfer", "trajectory": [()]},
                    {"phase": "post_place_lift", "trajectory": [()]},
                ]
            return self.get_observation(), TAMPFeedback(
                motion_planner_feedback="Success",
                action_success=True,
            )

    env = FakeEnv()
    saved_states = {}
    next_state_id = [0]

    def save_state():
        next_state_id[0] += 1
        saved_states[next_state_id[0]] = list(env.state)
        return next_state_id[0]

    monkeypatch.setattr(main_foresight.p, "saveState", save_state)
    monkeypatch.setattr(
        main_foresight.p,
        "restoreState",
        lambda state_id: setattr(env, "state", list(saved_states[state_id])),
    )
    monkeypatch.setattr(main_foresight.p, "removeState", lambda _state_id: None)

    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = env
    planner.use_acfm = True
    planner.candidate_selection_mode = "top1"
    planner._last_planning_llm_calls = 0
    planner._task_type = "packing"
    planner._packing_slot_strategy = None
    planner._non_llm_action_skeleton = lambda _config: [
        primitives["pick"](["box"], {}),
        primitives["place"](["box"], {}),
    ]
    planner._prepare_sequential_packing_strategy = lambda _actions: None
    fake_map = object()
    planner._compute_map_candidates = lambda *_args: (
        fake_map,
        [
            {
                "candidate_id": "",
                "object": "box",
                "x": 0.3,
                "y": 0.1,
                "theta": 0.0,
                "contact_mode": "top",
                "score": 0.95,
                "hard_constraints": {"bounds": True},
                "hard_feasible": True,
            },
            {
                "candidate_id": "",
                "object": "box",
                "x": 0.2,
                "y": 0.1,
                "theta": 0.0,
                "contact_mode": "top",
                "score": 0.9,
                "hard_constraints": {"bounds": True},
                "hard_feasible": True,
            },
        ],
    )

    def validate(template, candidate, _next):
        candidate["hard_constraints"]["transition"] = True
        candidate["endpoint_map_feasible"] = True
        candidate["ground_truth_feasible"] = candidate["x"] == 0.2
        candidate["hard_feasible"] = candidate["ground_truth_feasible"]
        candidate["validation_feedback"] = (
            "Success" if candidate["hard_feasible"] else "retreat failed")
        if candidate["hard_feasible"]:
            candidate["_validated_action"] = planner._resolve_candidate_action(
                template, candidate)
        return candidate

    planner._validate_candidate_rollout = validate
    history = []

    plan = planner._plan_sequentially_with_acfm(
        obs_text="box on table",
        feedback_list=[],
        task_config={},
        acfm_history=history,
        iteration=0,
    )

    assert [action.primitive.name for action in plan] == ["pick", "place"]
    assert plan[1].param_args == {"x": 0.2, "y": 0.1, "theta": 0.0}
    assert env.state == []  # predicted rollout is restored, never committed
    assert len(history) == 2
    assert history[0]["state"]["observation"]["events"] == []
    assert history[0]["state_after"]["observation"]["events"] == ["pick"]
    assert history[1]["state"]["observation"]["events"] == ["pick"]
    assert history[1]["state_after"]["observation"]["events"] == [
        "pick", "place"
    ]
    assert history[1]["map"] is fake_map
    assert history[1]["decision"]["candidate_id"] == "A02C002"
    assert [row["candidate_id"] for row in history[1]["candidate_rows"]] == [
        "A02C002"
    ]
    assert [
        row["ground_truth_feasible"]
        for row in history[1]["candidate_outcomes"]
    ] == [False, True]
    assert history[1]["candidate_outcomes"][0]["validation_feedback"] == (
        "retreat failed"
    )
    assert [row["status"] for row in history] == ["advanced", "advanced"]
    assert planner._last_planning_llm_calls == 0


def test_persistent_history_is_json_safe_and_detached_between_attempts():
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.acfm_hist = []
    map_grid = np.array([[0.0, 0.8], [0.5, 1.0]])
    fake_map = SimpleNamespace(
        target_name="box",
        res=2,
        z_res=None,
        bounds={"x_min": np.float64(0.1), "x_max": 0.2},
        x_range=np.array([0.1, 0.2]),
        y_range=np.array([-0.1, 0.1]),
        z_range=None,
        grid=map_grid,
        grid3d=None,
        hard_masks={"bounds": np.array([[False, True], [True, True]])},
        feasible_mask=np.array([[False, True], [True, True]]),
        soft_layers={"reachability": map_grid},
        contact_modes=("top",),
        parameter_candidates=[],
    )
    first_attempt = [{
        "iteration": 0,
        "step": 0,
        "status": "advanced",
        "state": {"observation": {"pose": np.array([0.1, 0.2])}},
        "state_after": {"observation": {"pose": np.array([0.2, 0.3])}},
        "map": fake_map,
        "acfm": fake_map,
        "object": "box",
        "action": "place",
        "candidate_rows": [],
        "candidate_outcomes": [],
        "decision": {"candidate_id": "A01C001", "parameters": {}},
    }]

    planner._persist_attempt_history(first_attempt)
    map_grid[0, 1] = 0.1
    first_attempt.clear()
    planner._persist_attempt_history([{
        "iteration": 1,
        "step": 0,
        "status": "no_hard_feasible_candidate",
        "state": {"observation": {}},
        "state_after": None,
        "map": None,
        "acfm": None,
        "object": "box",
        "action": "place",
        "candidate_rows": [],
        "candidate_outcomes": [],
        "decision": {"candidate_id": None, "parameters": {}},
    }])

    assert len(planner.acfm_hist) == 2
    assert planner.acfm_hist[0]["map"]["scores"]["xy"][0][1] == 0.8
    assert planner.acfm_hist[1]["map"] is None
    assert "acfm" not in planner.acfm_hist[0]
    json.dumps(planner.acfm_hist)


def test_failed_candidate_set_is_retained_in_attempt_history(monkeypatch):
    primitives = _primitives()

    class FakeEnv:
        primitive_actions = primitives
        objects = {"box": 1}
        theta_dict = {}
        robot = SimpleNamespace(
            attachments_robot=[],
            last_grasp_direction=None,
            last_grasp_position_offset=None,
        )

        def get_observation(self):
            return {"state": "unchanged"}

    env = FakeEnv()
    monkeypatch.setattr(main_foresight.p, "saveState", lambda: 1)
    monkeypatch.setattr(main_foresight.p, "restoreState", lambda _state: None)
    monkeypatch.setattr(main_foresight.p, "removeState", lambda _state: None)

    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = env
    planner.use_acfm = True
    planner.candidate_selection_mode = "top1"
    planner._last_planning_llm_calls = 0
    planner._task_type = "packing"
    planner._packing_slot_strategy = None
    planner._non_llm_action_skeleton = lambda _config: [
        primitives["place"](["box"], {})
    ]
    planner._prepare_sequential_packing_strategy = lambda _actions: None
    planner._compute_map_candidates = lambda *_args: (
        object(),
        [{
            "object": "box",
            "x": 0.2,
            "y": 0.1,
            "theta": 0.0,
            "contact_mode": "top",
            "score": 0.9,
            "hard_constraints": {"bounds": True},
            "hard_feasible": True,
        }],
    )

    def reject(_template, candidate, _next):
        candidate["endpoint_map_feasible"] = True
        candidate["ground_truth_feasible"] = False
        candidate["hard_constraints"]["retreat"] = False
        candidate["hard_feasible"] = False
        candidate["validation_feedback"] = "retreat failed"
        return candidate

    planner._validate_candidate_rollout = reject
    history = []

    plan = planner._plan_sequentially_with_acfm(
        obs_text="box held",
        feedback_list=[],
        task_config={},
        acfm_history=history,
        iteration=3,
    )

    assert plan is None
    assert len(history) == 1
    assert history[0]["status"] == "no_hard_feasible_candidate"
    assert history[0]["candidate_rows"] == []
    assert history[0]["candidate_outcomes"] == [{
        "candidate_id": "A01C001",
        "object": "box",
        "x": 0.2,
        "y": 0.1,
        "theta": 0.0,
        "contact_mode": "top",
        "score": 0.9,
        "hard_constraints": {"bounds": True, "retreat": False},
        "endpoint_map_feasible": True,
        "ground_truth_feasible": False,
        "validation_feedback": "retreat failed",
    }]


def test_counterfactual_suffix_replay_uses_checkpoint_and_restores_state(
        monkeypatch):
    place = _primitives()["place"]

    class FakeEnv:
        def __init__(self):
            self.state = []
            self.theta_dict = {}
            self.robot = SimpleNamespace(
                attachments_robot=[],
                last_grasp_direction="top",
                last_grasp_position_offset=[0.0, 0.0, 0.0],
            )

        def step(self, action, play_traj=False):
            self.state.append((action.obj_args[0], action.param_args["x"]))
            success = (
                action.obj_args[0] == "cause"
                or self.state[-2] == ("cause", 0.5)
            )
            return None, TAMPFeedback(
                motion_planner_feedback=("Success" if success else "blocked"),
                action_success=success,
            )

    env = FakeEnv()
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = env
    saved_states = {}
    next_id = [0]

    def save_state():
        next_id[0] += 1
        saved_states[next_id[0]] = list(env.state)
        return next_id[0]

    monkeypatch.setattr(main_foresight.p, "saveState", save_state)
    monkeypatch.setattr(
        main_foresight.p,
        "restoreState",
        lambda state_id: setattr(env, "state", list(saved_states[state_id])),
    )
    monkeypatch.setattr(main_foresight.p, "removeState", lambda _state: None)

    cause_state_id = save_state()
    checkpoint = {
        "state_id": cause_state_id,
        "python_state": planner._snapshot_python_sim_state(env),
    }
    env.state = [("committed_failure_state", -1.0)]
    plan = [
        place(["cause"], {"x": 0.1, "y": 0.0, "theta": 0.0}),
        place(["symptom"], {"x": 0.9, "y": 0.0, "theta": 0.0}),
    ]

    result = planner._replay_counterfactual_suffix(
        plan=plan,
        failed_step=1,
        checkpoints=[checkpoint],
        cause_step=0,
        alternative={"candidate_id": "A01C002", "x": 0.5, "y": 0.0},
        suffix=[str(plan[1])],
        failed_action=plan[1],
    )

    assert result == {"success": True}
    assert env.state == [("committed_failure_state", -1.0)]
