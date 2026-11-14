from types import SimpleNamespace

import numpy as np
import pytest

import main_foresight
from generate_results import (
    ABLATION_METHODS,
    METHODS,
    protocol_method_config,
)
from main_foresight import (
    DEFAULT_ROLLOUT_VALIDATION_QUOTAS,
    ForeSightTAMP,
)
from utils.metrics_util import trial_trace
from utils.tamp_util import Parameter, PrimitiveAction, TAMPFeedback


def _place_primitive():
    return PrimitiveAction(
        "place",
        obj_arity=1,
        parameters={
            "x": Parameter("x", -1.0, 1.0),
            "y": Parameter("y", -1.0, 1.0),
            "theta": Parameter("theta", -3.14, 3.14),
        },
    )


def test_rollout_control_samples_raw_domain_without_scores_or_masks():
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = SimpleNamespace()
    planner.rollout_validation_quotas = dict(
        DEFAULT_ROLLOUT_VALIDATION_QUOTAS)
    template = _place_primitive()(["box"], {})
    mapper = SimpleNamespace(
        x_range=np.linspace(0.1, 0.4, 4),
        y_range=np.linspace(-0.2, 0.2, 5),
        target_dims=[0.20, 0.08, 0.05],
        contact_modes=("top",),
        # These fields must have no effect on control candidates.
        grid=np.arange(20, dtype=float).reshape(5, 4),
        feasible_mask=np.zeros((5, 4), dtype=bool),
        hard_masks={"bounds": np.zeros((5, 4), dtype=bool)},
    )

    first, accounting = planner._rollout_control_candidates(
        mapper, template, reference_candidates=[])
    mapper.grid[:] = -999.0
    mapper.feasible_mask[:] = True
    mapper.hard_masks["bounds"][:] = True
    second, repeated_accounting = planner._rollout_control_candidates(
        mapper, template, reference_candidates=[])
    next_batch, next_accounting = planner._rollout_control_candidates(
        mapper, template, reference_candidates=[], batch_index=1)

    assert first == second
    assert accounting == repeated_accounting
    assert len(first) == DEFAULT_ROLLOUT_VALIDATION_QUOTAS["place"]
    assert accounting == {
        "selection_mode": "rollout_control",
        "candidate_source": "score_independent_low_discrepancy_domain",
        "domain_size": 40,
        "rollout_quota": DEFAULT_ROLLOUT_VALIDATION_QUOTAS["place"],
        "sampling_batch": 0,
    }
    assert all(row["_rollout_only"] for row in first)
    assert all(row["hard_constraints"] == {} for row in first)
    assert all("score" not in row for row in first)
    parameter_key = lambda row: (
        row["x"], row["y"], row["theta"], row["contact_mode"])
    assert not ({parameter_key(row) for row in first}
                & {parameter_key(row) for row in next_batch})
    assert next_accounting["sampling_batch"] == 1


def test_rollout_only_validation_does_not_require_endpoint_map_acceptance(
        monkeypatch):
    primitive = _place_primitive()

    class FakeEnv:
        def __init__(self):
            self.theta_dict = {}
            self.robot = SimpleNamespace(
                attachments_robot=[],
                last_grasp_direction=None,
                last_grasp_position_offset=None,
                last_motion_trace=[],
            )

        def step(self, _action, play_traj=False):
            self.robot.last_motion_trace = [
                {"phase": "place_transfer", "trajectory": [()]},
                {"phase": "post_place_lift", "trajectory": [()]},
            ]
            return None, TAMPFeedback(
                motion_planner_feedback="Success",
                action_success=True,
            )

    monkeypatch.setattr(main_foresight.p, "saveState", lambda: 1)
    monkeypatch.setattr(main_foresight.p, "restoreState", lambda _state: None)
    monkeypatch.setattr(main_foresight.p, "removeState", lambda _state: None)
    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = FakeEnv()
    planner.candidate_selection_mode = "rollout_control"
    candidate = {
        "candidate_id": "A01C001",
        "object": "box",
        "x": 0.2,
        "y": 0.1,
        "theta": 0.0,
        "hard_constraints": {},
        "hard_feasible": False,
        "_rollout_only": True,
    }

    with trial_trace(max_llm_requests=10) as trace:
        result = planner._validate_candidate_rollout(
            primitive(["box"], {}), candidate, next_template=None)

    assert result["endpoint_map_feasible"] is None
    assert result["ground_truth_feasible"] is True
    assert result["hard_feasible"] is True
    assert result["hard_constraints"] == {
        "transition": True,
        "approach_place": True,
        "retreat": True,
    }
    assert trace.summary()["candidate_rollout_attempts"] == 1
    assert trace.summary()["candidate_rollout_passes"] == 1
    assert trace.events[-1]["phase"] == "acfm_validation"


def test_rollout_control_selects_first_passing_id_and_audits_budget(monkeypatch):
    pick = PrimitiveAction("pick", obj_arity=1)
    place = _place_primitive()

    class FakeEnv:
        primitive_actions = {"pick": pick, "place": place}
        objects = {"box": 1}
        theta_dict = {}

        def __init__(self):
            self.state = []
            self.robot = SimpleNamespace(
                attachments_robot=[],
                last_grasp_direction=None,
                last_grasp_position_offset=None,
                last_motion_trace=[],
            )

        def get_observation(self):
            return {"state": list(self.state)}

        def step(self, action, play_traj=False):
            self.state.append(action.primitive.name)
            return self.get_observation(), TAMPFeedback(
                motion_planner_feedback="Success",
                action_success=True,
            )

    env = FakeEnv()
    saved_states = {}
    state_counter = [0]

    def save_state():
        state_counter[0] += 1
        saved_states[state_counter[0]] = list(env.state)
        return state_counter[0]

    monkeypatch.setattr(main_foresight.p, "saveState", save_state)
    monkeypatch.setattr(
        main_foresight.p,
        "restoreState",
        lambda state_id: setattr(env, "state", list(saved_states[state_id])),
    )
    monkeypatch.setattr(main_foresight.p, "removeState", lambda _state: None)

    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    planner.env = env
    planner.use_acfm = True
    planner.candidate_selection_mode = "rollout_control"
    planner.rollout_validation_quotas = dict(
        DEFAULT_ROLLOUT_VALIDATION_QUOTAS)
    planner._last_planning_llm_calls = 0
    planner._task_type = "packing"
    planner._packing_slot_strategy = None
    planner._non_llm_action_skeleton = lambda _config: [
        pick(["box"], {}), place(["box"], {})]
    planner._prepare_sequential_packing_strategy = lambda _actions: None
    planner._compute_map_candidates = lambda *_args: (object(), [{
        "score": 1.0,
        "hard_feasible": True,
    }])
    control_rows = [
        {
            "object": "box",
            "x": 0.1,
            "y": 0.0,
            "theta": 0.0,
            "hard_constraints": {},
            "hard_feasible": False,
            "_rollout_only": True,
        },
        {
            "object": "box",
            "x": 0.2,
            "y": 0.0,
            "theta": 0.0,
            "hard_constraints": {},
            "hard_feasible": False,
            "_rollout_only": True,
        },
    ]
    planner._rollout_control_candidates = lambda *_args, **_kwargs: (
        control_rows,
        {
            "selection_mode": "rollout_control",
            "candidate_source": "score_independent_low_discrepancy_domain",
            "domain_size": 40,
            "rollout_quota": 10,
        },
    )

    def validate(template, candidate, _next):
        candidate["endpoint_map_feasible"] = None
        candidate["ground_truth_feasible"] = candidate["x"] == 0.2
        candidate["hard_feasible"] = candidate["ground_truth_feasible"]
        candidate["hard_constraints"]["transition"] = candidate[
            "ground_truth_feasible"]
        candidate["validation_feedback"] = (
            "Success" if candidate["hard_feasible"] else "blocked")
        if candidate["hard_feasible"]:
            candidate["_validated_action"] = (
                planner._resolve_candidate_action(template, candidate))
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

    assert plan[1].param_args["x"] == 0.2
    assert planner._last_planning_llm_calls == 0
    assert history[1]["llm_candidate_rows"] == []
    assert history[1]["decision"]["candidate_id"] == "A02C002"
    assert history[1]["candidate_rollout_accounting"] == {
        "selection_mode": "rollout_control",
        "candidate_source": "score_independent_low_discrepancy_domain",
        "domain_size": 40,
        "rollout_quota": 10,
        "rollout_attempted": 2,
        "rollout_passed": 1,
    }
    assert env.state == []


@pytest.mark.parametrize("mode", ["top1", "rollout_control"])
def test_non_llm_modes_do_not_initialize_a_planner_or_read_credentials(
        tmp_path, monkeypatch, mode):
    class FakeEnv:
        primitive_actions = {}

        def reset(self, **_kwargs):
            return {}, "empty test world"

    def forbidden_init(*_args, **_kwargs):
        pytest.fail("non-LLM mode attempted to initialize GPT_Chat")

    monkeypatch.setattr(ForeSightTAMP, "_init_planner", forbidden_init)
    planner = ForeSightTAMP(
        env=FakeEnv(),
        max_llm_calls=0,
        save_dir=tmp_path / mode,
        candidate_selection_mode=mode,
    )
    result = planner.run_episode({
        "basket": {},
        "boxes": {},
        "task_type": "packing",
        "task": "unused",
    })

    assert planner.planner is None
    assert result["num_llm_calls"] == 0


def test_equal_rollout_method_is_frozen_as_execution_configuration():
    assert "Abl_matched_rollout" in ABLATION_METHODS
    assert METHODS["Abl_matched_rollout"]["candidate_selection_mode"] == (
        "rollout_control")
    assert protocol_method_config("Abl_matched_rollout") == {
        "class": "foresight",
        "disable": [],
        "candidate_selection_mode": "rollout_control",
        "rollout_validation_quotas": dict(DEFAULT_ROLLOUT_VALIDATION_QUOTAS),
    }


def test_matched_control_budget_equals_foresight_budget():
    """The control arm only controls for compute if the budgets are equal.

    `Abl_matched_rollout` exists to give a non-LLM selector the same simulator
    allowance ForeSight receives. If the two drift apart, the comparison
    silently rewards whichever side kept the larger budget.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "generate_results.py").read_text()
    block = re.search(
        r'"Abl_matched_rollout".*?"rollout_validation_quotas":\s*\{(.*?)\}',
        source, re.S)
    assert block is not None, "matched-rollout config not found"
    declared = {
        key: int(value)
        for key, value in re.findall(r'"(\w+)":\s*(\d+)', block.group(1))
    }
    assert declared == DEFAULT_ROLLOUT_VALIDATION_QUOTAS, (
        f"control budget {declared} differs from ForeSight's "
        f"{DEFAULT_ROLLOUT_VALIDATION_QUOTAS}")
