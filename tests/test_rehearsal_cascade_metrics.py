import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade
import rehearsal
from envs.pack_compact_env import PackCompactEnv
from envs.push_stacking_env import PushStackingEnv
from generate_results import _run_llm3
from utils.llm_util import connect_openai
from utils.metrics_util import LLMRequestBudgetExceeded, trial_trace
from utils.tamp_util import (
    Action,
    CachedTrajectory,
    PrimitiveAction,
    text_to_actions,
)


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _response(model="snapshot-model", prompt=11, completion=3):
    return SimpleNamespace(
        id="response-1",
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def test_llm_trace_counts_actual_retries_tokens_and_cost(monkeypatch):
    completions = _Completions([RuntimeError("transient"), _response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("utils.llm_util.time.sleep", lambda _: None)

    with trial_trace(
        max_llm_requests=2,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    ) as trace:
        response = connect_openai(
            client=client,
            engine="requested-alias",
            messages=[{"role": "user", "content": "prompt"}],
            temperature=0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format=None,
            max_transport_attempts=5,
        )

    assert response.model == "snapshot-model"
    summary = trace.summary()
    assert summary["llm_api_requests"] == 2
    assert summary["llm_transport_retries"] == 1
    assert summary["llm_prompt_tokens"] == 11
    assert summary["llm_completion_tokens"] == 3
    assert summary["llm_estimated_cost_usd"] == pytest.approx(17 / 1_000_000)
    assert [event["status"] for event in trace.llm_events] == [
        "failed",
        "succeeded",
    ]


def _call(client, **overrides):
    kwargs = dict(
        client=client, engine="model", messages=[], temperature=0,
        max_tokens=10, top_p=1, frequency_penalty=0, presence_penalty=0,
        response_format=None, max_transport_attempts=5,
    )
    kwargs.update(overrides)
    return connect_openai(**kwargs)


def test_a_transport_retry_does_not_consume_the_planning_budget(monkeypatch):
    """The budget bounds how often the planner asks, not how often the
    transport fails.

    Charging retries to the planner makes a result depend on server health: a
    CUDA fault on the serving side consumed five of ten calls and failed
    scenarios the model had already solved elsewhere.
    """
    completions = _Completions([RuntimeError("transient"), _response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("utils.llm_util.time.sleep", lambda _: None)

    with trial_trace(max_llm_requests=1) as trace:
        response = _call(client)

    assert response.model == "snapshot-model"
    summary = trace.summary()
    assert summary["llm_logical_calls"] == 1     # one question asked
    assert summary["llm_api_requests"] == 2      # two HTTP attempts
    assert summary["llm_transport_retries"] == 1
    assert summary["llm_budget_rejections"] == 0


def test_llm_trace_enforces_hard_request_budget(monkeypatch):
    """A second logical call beyond the budget is still refused."""
    completions = _Completions([_response(), _response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("utils.llm_util.time.sleep", lambda _: None)

    with trial_trace(max_llm_requests=1) as trace:
        _call(client)
        with pytest.raises(LLMRequestBudgetExceeded):
            _call(client)

    assert completions.calls == 1
    assert trace.summary()["llm_budget_rejections"] == 1


def test_rehearsal_caches_exact_trajectory_and_restores_python_state(monkeypatch):
    primitive = PrimitiveAction("pick", obj_arity=1)
    action = Action(primitive, ["box"], {}, [])

    class Env:
        theta_dict = {"box": 0.2}
        objects = {}
        robot = SimpleNamespace(
            attachments_robot=["original"], last_grasp_direction="top"
        )

        def step(self, candidate, play_traj=False):
            candidate.traj = [(0.0, 0.1), (0.2, 0.3)]
            self.robot.attachments_robot = ["mutated"]
            self.robot.last_grasp_direction = "side"
            self.theta_dict["box"] = 1.0
            return None, SimpleNamespace(
                action_success=True,
                goal_achieved=False,
                motion_planner_feedback="",
            )

    monkeypatch.setattr(rehearsal.p, "saveState", lambda: 7)
    monkeypatch.setattr(rehearsal.p, "restoreState", lambda state: None)
    monkeypatch.setattr(rehearsal.p, "removeState", lambda state: None)

    env = Env()
    engine = rehearsal.MentalRehearsal(env)
    result = rehearsal.MentalRehearsal.rehearse.__wrapped__(engine, [action])

    assert result.success
    assert result.motion_planner_calls == 1
    assert result.cached_trajectories == [[(0.0, 0.1), (0.2, 0.3)]]
    assert env.robot.attachments_robot == ["original"]
    assert env.robot.last_grasp_direction == "top"
    assert env.theta_dict == {"box": 0.2}

    action.traj = []
    engine.apply_cached_trajectories([action], result)
    assert action.traj == [(0.0, 0.1), (0.2, 0.3)]


def test_cached_replay_preserves_pick_and_place_state_change_boundaries():
    scenario_path = (
        Path(__file__).parents[1]
        / "tests"
        / "fixtures"
        / "PKG2_tomato_cans_6_box.json"
    )
    with scenario_path.open() as stream:
        scenario = json.load(stream)

    env = PackCompactEnv()
    try:
        env.reset(
            basket=scenario["basket"],
            boxes=scenario["boxes"],
            task_type="packing",
            use_gui=False,
        )
        plan = text_to_actions(
            [
                "pick(['tomato_soup_can_1'], {})",
                (
                    "place(['tomato_soup_can_1'], "
                    "{'x': 0.357, 'y': -0.065, 'theta': 0.0})"
                ),
            ],
            env.primitive_actions,
        )
        result = rehearsal.MentalRehearsal(env).rehearse(plan)
        assert result.success
        assert result.motion_planner_calls >= 5
        rehearsal.MentalRehearsal.apply_cached_trajectories(plan, result)

        observed_boundaries = []
        original_simulate_traj = env.robot.simulate_traj

        def observe_boundary(*args, **kwargs):
            observed_boundaries.append(
                (
                    env.robot.last_motion_trace[-1]["phase"],
                    bool(env.robot.attachments_robot),
                )
            )
            return original_simulate_traj(*args, **kwargs)

        env.robot.simulate_traj = observe_boundary
        calls_before_replay = env.robot.motion_planner_call_count
        for action in plan:
            _, feedback = env.step(action, play_traj=False)
            assert feedback.action_success

        assert env.robot.motion_planner_call_count == calls_before_replay
        assert observed_boundaries == [
            ("pick_pregrasp", False),
            ("pick_cartesian_descent", False),
            ("post_grasp_cartesian_lift", True),
            ("place_transfer", True),
            ("post_place_lift", False),
        ]
    finally:
        env.destroy()


def test_cached_push_trace_binds_full_transition_contact_and_retreat_path():
    segments = [
        {"phase": "current_to_push_contact", "trajectory": [(0.0,), (0.1,)]},
        {"phase": "push_contact", "trajectory": [(0.1,), (0.2,)]},
        {"phase": "post_push_retreat", "trajectory": [(0.3,), (0.4,)]},
    ]
    trajectory = CachedTrajectory(
        [(0.0,), (0.1,), (0.1,), (0.2,), (0.3,), (0.4,)],
        segments,
    )
    validated = PushStackingEnv._validated_cached_push_segments(trajectory)
    assert [phase for phase, _ in validated] == [
        "current_to_push_contact",
        "push_contact",
        "post_push_retreat",
    ]

    trajectory[-1] = (9.0,)
    with pytest.raises(ValueError, match="do not match"):
        PushStackingEnv._validated_cached_push_segments(trajectory)


class _ACFM:
    def get_best_positions(self, count):
        return [{"x": 0.4, "y": 0.2, "score": 0.9}][:count]


def test_cascade_uses_most_recent_writer_and_emits_valid_json():
    plan = [
        "place(['blue_box'], {'x': 0.1, 'y': 0.1})",
        "place(['red_box'], {'x': 0.2, 'y': 0.2})",
        "place(['blue_box'], {'x': 0.3, 'y': 0.3})",
        "place(['red_box'], {'x': 0.4, 'y': 0.4})",
    ]
    history = [{
        "step": 2,
        "object": "blue_box",
        "decision": {"candidate_id": "A03C001"},
        "candidate_rows": [
            {
                "candidate_id": "A03C001",
                "x": 0.3,
                "y": 0.3,
                "theta": 0.0,
                "score": 0.81,
                "hard_constraints": {"reachable": True},
            },
            {
                "candidate_id": "A03C002",
                "x": 0.4,
                "y": 0.2,
                "theta": 1.57,
                "score": 0.9,
                "hard_constraints": {"reachable": True},
            },
        ],
    }]
    replays = []

    def replay(**kwargs):
        replays.append(kwargs)
        return {"success": True}

    analyzer = cascade.CascadeAnalyzer(
        env=object(), acfm_history=history, counterfactual_replayer=replay
    )
    result = analyzer.analyze(
        failed_action=plan[-1],
        failed_step=3,
        failure_feedback="collision with object blue_box",
        executed_plan=plan,
        placed_positions={"blue_box": {"x": 0.3, "y": 0.3}},
    )

    assert result.root_cause_step == 2
    assert result.counterfactual_confirmed is True
    assert result.attribution_status == "counterfactually-confirmed"
    assert replays[0]["cause_step"] == 2
    assert replays[0]["alternative"]["candidate_id"] == "A03C002"
    assert replays[0]["alternative"]["x"] == 0.4
    report = json.loads(analyzer.format_report_for_llm(result))
    assert report["suspected_root_step"] == 2
    assert report["indexing"] == "zero_based"
    assert any(
        edge["relation"] == "observed-collision-blocker"
        for edge in report["provenance_graph"]["edges"]
    )
    assert report["blocking_chain"][0]["best_alternative"] == {
        "candidate_id": "A03C002",
        "score": 0.9,
    }

    def all_keys(value):
        if isinstance(value, dict):
            return set(value).union(*(all_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(all_keys(item) for item in value))
        return set()

    assert not {"x", "y", "z", "theta", "position"} & all_keys(report)
    assert "0.400" not in report["suggestion"]


def test_discarded_primitive_metric_excludes_failed_action():
    pick = PrimitiveAction("pick", obj_arity=1)

    class Planner:
        def __init__(self):
            self.calls = 0

        def plan(self, *_args, **_kwargs):
            self.calls += 1
            return [Action(pick, ["box"]), Action(pick, ["box"])]

    class Env:
        def __init__(self):
            self.step_index = 0

        def reset(self, **_kwargs):
            self.step_index = 0
            return None, "observation"

        def get_symbolic_plan(self):
            return []

        def step(self, action, play_traj=False):
            self.step_index += 1
            # First action changes state; second fails before execution.
            return None, SimpleNamespace(
                action_success=self.step_index == 1,
                goal_achieved=False,
            )

    result = _run_llm3(
        Env(), Planner(), {}, max_llm_calls=2, use_gui=False, play_traj=False
    )
    assert result["discarded_simulator_primitives"] == 2
    assert result["committed_successful_simulator_primitives"] == 2
