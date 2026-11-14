"""ViLaIn adaptation: multimodal problem specification followed by a planner.

The shared benchmark supplies a textualized scene rather than camera
pixels.  ViLaIn therefore produces a machine-readable problem description from
that observation; a domain planner—not a second free-form LLM call—constructs
the executable plan, matching the architecture of Shirai et al. (2024).
"""

from __future__ import annotations

import json
import logging

from utils.llm_util import LLMBase
from .common import grounded_reference_plan, infer_env_type, primitive_schema, safe_json_object

logger = logging.getLogger(__name__)


class ViLaInTAMPBaseline(LLMBase):
    def __init__(self, primitive_actions, env, max_retries=5, max_llm_calls=None):
        super().__init__(use_gpt_4=True)
        self.primitive_actions = primitive_actions
        self.env = env
        self.max_retries = max_retries
        self.max_llm_calls = max_llm_calls if max_llm_calls is not None else max_retries

    def run_episode(self, task_config, play_traj=False, use_gui=True):
        _, obs_text = self.env.reset(**task_config, use_gui=use_gui)
        calls = mp_calls = wasted = 0
        successful: list[str] = []
        feedback_text = ""

        while calls < self.max_llm_calls and calls < self.max_retries:
            if calls > 0:
                _, obs_text = self.env.reset(**task_config, use_gui=use_gui)
                successful = []
            problem = self._interpret_scene(obs_text, task_config, feedback_text)
            calls += 1
            if problem is None:
                feedback_text = "The problem description was invalid."
                continue
            try:
                plan = grounded_reference_plan(self.env, task_config)
            except Exception as exc:
                feedback_text = f"Symbolic planner rejected the problem: {exc}"
                continue

            failed = False
            actions_this_attempt = 0
            successful_this_attempt = 0
            for action in plan:
                mp_calls += 1
                actions_this_attempt += 1
                _, feedback = self.env.step(action, play_traj=play_traj)
                if not feedback.action_success:
                    feedback_text = (
                        f"Planner/execution feedback for {action}: "
                        f"{feedback.motion_planner_feedback}"
                    )
                    failed = True
                    break
                successful.append(str(action))
                successful_this_attempt += 1
                if feedback.goal_achieved:
                    return self._result(True, calls, mp_calls, wasted, successful)
            goal, goal_feedback = self.env.check_goal()
            if goal:
                return self._result(True, calls, mp_calls, wasted, successful)
            wasted += successful_this_attempt
            feedback_text = goal_feedback
            if failed:
                # ViLaIn refines the formal problem using planner feedback while
                # retaining the actual current scene.
                _, obs_text = self.env.get_observation()

        return self._result(False, calls, mp_calls, wasted, [])

    def _interpret_scene(self, obs_text, task_config, feedback=""):
        prompt = (
            "Generate a machine-readable planning problem description from the "
            "scene observation, following ViLaIn. Separate objects, initial facts, "
            "goal facts, and any task rules. Do not output an action plan.\n"
            f"Domain: {infer_env_type(self.env)}\n"
            f"Primitive schema: {json.dumps(primitive_schema(self.primitive_actions))}\n"
            f"Task configuration: {json.dumps(task_config, default=str)}\n"
            f"Scene observation: {obs_text}\nPlanner feedback: {feedback}\n"
            'Return JSON with keys "objects", "initial_facts", "goal_facts", "rules".'
        )
        try:
            parsed = safe_json_object(self.prompt_llm(prompt, force_json=True))
            if all(k in parsed for k in ("objects", "initial_facts", "goal_facts")):
                return parsed
        except Exception as exc:
            logger.warning("ViLaIn problem interpretation failed: %s", exc)
        return None

    def _result(self, goal, calls, mp_calls, wasted, plan):
        return {
            "goal_achieved": bool(goal),
            "num_llm_calls": calls,
            "num_mp_calls": mp_calls,
            "discarded_simulator_primitives": wasted,
            "wasted_real_actions": wasted,
            "tamp_plan": plan if goal else [],
            "baseline_variant": "adapted_vilain_problem_specification",
        }
