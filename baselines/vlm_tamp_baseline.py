"""VLM-TAMP adaptation with intermediate-goal generation and TAMP refinement."""

from __future__ import annotations

import json
import logging

from utils.llm_util import LLMBase
from .common import grounded_reference_plan, infer_env_type, safe_json_object

logger = logging.getLogger(__name__)


class VLMTAMPBaseline(LLMBase):
    def __init__(self, primitive_actions, env, max_subgoals=20,
                 grid_resolution=8, max_llm_calls=None):
        super().__init__(use_gpt_4=True)
        self.primitive_actions = primitive_actions
        self.env = env
        self.max_subgoals = max_subgoals
        self.max_llm_calls = max_llm_calls if max_llm_calls is not None else max_subgoals

    def run_episode(self, task_config, play_traj=False, use_gui=True):
        _, obs_text = self.env.reset(**task_config, use_gui=use_gui)
        reference = grounded_reference_plan(self.env, task_config)
        cursor = calls = mp_calls = wasted = 0
        successful: list[str] = []
        refinement_feedback = ""

        while cursor < len(reference) and calls < self.max_llm_calls:
            subgoals = self._generate_subgoals(
                obs_text, reference[cursor:], refinement_feedback)
            calls += 1
            if not subgoals:
                refinement_feedback = "No valid intermediate goals were returned."
                continue

            made_progress = False
            for subgoal in subgoals[:self.max_subgoals]:
                obj = str(subgoal.get("object", ""))
                # The classical TAMP layer refines an intermediate goal into the
                # next primitive segment. A pick/place segment is kept atomic.
                if cursor >= len(reference):
                    break
                segment = [reference[cursor]]
                if reference[cursor].primitive.name == "pick" and cursor + 1 < len(reference):
                    segment.append(reference[cursor + 1])
                expected_obj = segment[0].obj_args[0]
                if obj != expected_obj:
                    refinement_feedback = (
                        f"Subgoal for {obj!r} cannot be refined now; "
                        f"the current symbolic state requires {expected_obj!r}."
                    )
                    break

                segment_ok = True
                for action in segment:
                    mp_calls += 1
                    _, feedback = self.env.step(action, play_traj=play_traj)
                    if not feedback.action_success:
                        # A failed motion plan does not execute a simulator
                        # primitive. Clear any partial cached path before retry.
                        action.traj = []
                        refinement_feedback = (
                            f"Failed to refine {subgoal}: "
                            f"{feedback.motion_planner_feedback}"
                        )
                        segment_ok = False
                        break
                    successful.append(str(action))
                    cursor += 1
                    made_progress = True
                    if feedback.goal_achieved:
                        return self._result(True, calls, mp_calls, wasted, successful)
                if not segment_ok:
                    break
            _, obs_text = self.env.get_observation()
            if not made_progress and not refinement_feedback:
                refinement_feedback = "The proposed subgoal sequence made no progress."

        goal, _ = self.env.check_goal()
        if not goal:
            # Progressive subgoals are retained across planner calls; they are
            # discarded only when the episode ultimately fails.
            wasted = len(successful)
        return self._result(goal, calls, mp_calls, wasted, successful if goal else [])

    def _generate_subgoals(self, obs_text, remaining_plan, feedback):
        candidate_objects = sorted({a.obj_args[0] for a in remaining_plan if a.obj_args})
        prompt = (
            "Act as the VLM high-level component of VLM-TAMP. Generate a short "
            "ordered sequence of horizon-reducing intermediate goals. Do not emit "
            "robot trajectories or continuous coordinates; a classical TAMP solver "
            "will refine each goal. Use only objects visible in the scene.\n"
            f"Domain: {infer_env_type(self.env)}\nScene:\n{obs_text}\n"
            f"Objects that the TAMP problem still permits: {json.dumps(candidate_objects)}\n"
            f"Previous refinement feedback: {feedback}\n"
            'Return JSON: {"subgoals":[{"object":"name","relation":"goal predicate"}]}'
        )
        try:
            parsed = safe_json_object(self.prompt_llm(prompt, force_json=True))
            goals = parsed.get("subgoals", [])
            if isinstance(goals, list):
                return [g for g in goals if isinstance(g, dict) and g.get("object")]
        except Exception as exc:
            logger.warning("VLM-TAMP subgoal generation failed: %s", exc)
        return []

    def _result(self, goal, calls, mp_calls, wasted, plan):
        return {
            "goal_achieved": bool(goal),
            "num_llm_calls": calls,
            "num_mp_calls": mp_calls,
            "discarded_simulator_primitives": wasted,
            "wasted_real_actions": wasted,
            "tamp_plan": plan,
            "baseline_variant": "adapted_vlm_tamp_hierarchical_refinement",
        }
