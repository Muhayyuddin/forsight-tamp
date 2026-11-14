"""OWL-TAMP adaptation using VLM-inferred declarative constraints.

Model output is never evaluated as Python. The VLM supplies discrete/continuous
constraints and the shared domain sampler constructs a plan satisfying those
constraints before physical execution.
"""

from __future__ import annotations

import json
import logging

from utils.llm_util import LLMBase
from .common import grounded_reference_plan, infer_env_type, safe_json_object

logger = logging.getLogger(__name__)


class OWLTAMPBaseline(LLMBase):
    def __init__(self, primitive_actions, env, max_retries=5,
                 grid_resolution=8, max_llm_calls=None):
        super().__init__(use_gpt_4=True)
        self.primitive_actions = primitive_actions
        self.env = env
        self.max_retries = max_retries
        self.max_llm_calls = max_llm_calls if max_llm_calls is not None else max_retries

    def run_episode(self, task_config, play_traj=False, use_gui=True):
        _, obs_text = self.env.reset(**task_config, use_gui=use_gui)
        calls = mp_calls = wasted = 0
        successful: list[str] = []
        feedback_context = ""

        while calls < min(self.max_retries, self.max_llm_calls):
            if calls > 0:
                _, obs_text = self.env.reset(**task_config, use_gui=use_gui)
                successful = []
            constraints = self._infer_constraints(obs_text, task_config, feedback_context)
            calls += 1
            if constraints is None:
                feedback_context = "Constraint generation did not match the schema."
                continue
            try:
                plan = grounded_reference_plan(self.env, task_config)
            except Exception as exc:
                feedback_context = f"Constraint solver failed: {exc}"
                continue

            # Constraint validation occurs before execution. Domain environments
            # remain authoritative for support, ordering, and legal-move rules.
            invalid = self._validate_constraints(constraints)
            if invalid:
                feedback_context = invalid
                continue

            actions_this_attempt = 0
            successful_this_attempt = 0
            for action in plan:
                mp_calls += 1
                actions_this_attempt += 1
                _, feedback = self.env.step(action, play_traj=play_traj)
                if not feedback.action_success:
                    feedback_context = (
                        f"Grounded constraint failure for {action}: "
                        f"{feedback.motion_planner_feedback}"
                    )
                    break
                successful.append(str(action))
                successful_this_attempt += 1
                if feedback.goal_achieved:
                    return self._result(True, calls, mp_calls, wasted, successful)
            goal, goal_feedback = self.env.check_goal()
            if goal:
                return self._result(True, calls, mp_calls, wasted, successful)
            wasted += successful_this_attempt
            feedback_context = goal_feedback
            _, obs_text = self.env.get_observation()

        return self._result(False, calls, mp_calls, wasted, [])

    def _infer_constraints(self, obs_text, task_config, feedback=""):
        prompt = (
            "Act as OWL-TAMP's vision-language constraint generator. Infer "
            "discrete temporal constraints and continuous spatial constraints for "
            "the open-world objective. Do not output executable code or coordinates "
            "that are not supported by the scene.\n"
            f"Domain: {infer_env_type(self.env)}\n"
            f"Task: {json.dumps(task_config, default=str)}\nScene:\n{obs_text}\n"
            f"Solver feedback: {feedback}\n"
            'Return JSON: {"temporal":[{"before":"object/action","after":"object/action"}],'
            '"spatial":[{"object":"name","relation":"inside/on/at","target":"name"}],'
            '"rules":["natural-language invariant"]}'
        )
        try:
            parsed = safe_json_object(self.prompt_llm(prompt, force_json=True))
            if isinstance(parsed.get("temporal", []), list) and isinstance(
                    parsed.get("spatial", []), list):
                return parsed
        except Exception as exc:
            logger.warning("OWL-TAMP constraint generation failed: %s", exc)
        return None

    def _validate_constraints(self, constraints):
        if not any(constraints.get(key) for key in ("temporal", "spatial", "rules")):
            return "Constraint generator returned no constraints."
        known = set(self.env.objects)
        for item in constraints.get("spatial", []):
            if not isinstance(item, dict):
                return "Spatial constraint is not an object."
            obj = item.get("object")
            if obj and obj not in known:
                return f"Unknown object in spatial constraint: {obj}"
        return ""

    def _result(self, goal, calls, mp_calls, wasted, plan):
        return {
            "goal_achieved": bool(goal),
            "num_llm_calls": calls,
            "num_mp_calls": mp_calls,
            "discarded_simulator_primitives": wasted,
            "wasted_real_actions": wasted,
            "tamp_plan": plan if goal else [],
            "baseline_variant": "adapted_owl_tamp_inferred_constraints",
        }

    def _parse_constraint(self, _code):
        raise ValueError("OWL-TAMP now accepts declarative JSON constraints, not Python code")
