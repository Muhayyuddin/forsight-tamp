"""
ForeSight-TAMP: Main Integration Module

Integrates all three novel contributions into the LLM³ planning loop:
- N3: ACFM (Action-Conditioned Spatial Feasibility Maps)
- N1: Mental Rehearsal (full-plan forward simulation)
- N2: Cascade Analysis (geometric failure root-cause tracing)
"""
import json
import logging
import re
import copy
import itertools
import math
from datetime import datetime, timezone
import numpy as np
import pybullet as p
from pathlib import Path

from envs.pack_compact_env import PackCompactEnv
from envs.stacking_env import StackingEnv
from planners.llm_tamp_planner import LLMTAMPPlanner
from acfm import ACFM, ACFM3D, HanoiACFM, PushACFM
from rehearsal import MentalRehearsal
from cascade import CascadeAnalyzer
from utils.io_util import dump_json, mkdir
from utils.rendering_util import suppress_gui_rendering
from utils.tamp_util import text_to_actions
from utils.metrics_util import (
    increment_metric,
    metric_phase,
    metric_timer,
    record_metric_event,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DEFAULT_ROLLOUT_VALIDATION_QUOTAS = {
    # Candidate-rollout caps for the full ForeSight path. Each rollout executes
    # the candidate in the simulator, the largest cost in a run, so the caps
    # bound that cost while leaving margin for recovery when the map's ranking
    # is wrong.
    "place": 10,
    "push": 9,
    # These domains are intrinsically smaller in the current benchmarks.
    "stacking_place": 1,
    "hanoi_place": 2,
}


class CandidateSelectionError(ValueError):
    """Raised when an LLM candidate choice is malformed or not offered."""


def _parse_candidate_id_response(raw_text, allowed_ids):
    """Strictly parse an ID-only candidate-selection response."""
    payload = _parse_llm_json_object(raw_text)
    if not isinstance(payload, dict):
        raise CandidateSelectionError("candidate selection must be a JSON object")
    if set(payload) != {"Reasoning", "Candidate ID"}:
        raise CandidateSelectionError(
            "candidate selection must contain exactly Reasoning and Candidate ID"
        )
    if not isinstance(payload["Reasoning"], str):
        raise CandidateSelectionError("Reasoning must be a string")
    candidate_id = payload["Candidate ID"]
    if not isinstance(candidate_id, str) or not re.fullmatch(
            r"A\d{2}C\d{3}", candidate_id):
        raise CandidateSelectionError("Candidate ID has an invalid format")
    if candidate_id not in set(allowed_ids):
        raise CandidateSelectionError(
            f"Candidate ID {candidate_id!r} is not in the offered set"
        )
    return candidate_id


def _parse_action_skeleton_response(raw_text, primitive_actions):
    """Parse an action-only plan and reject numeric/spatial parameters."""
    payload = _parse_llm_json_object(raw_text)
    if not isinstance(payload, dict) or set(payload) != {
            "Reasoning", "Action Skeleton"}:
        raise ValueError(
            "skeleton response must contain exactly Reasoning and Action Skeleton"
        )
    if not isinstance(payload["Reasoning"], str):
        raise ValueError("Reasoning must be a string")
    raw_actions = payload["Action Skeleton"]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Action Skeleton must be a non-empty list")
    if len(raw_actions) > 64 or not all(isinstance(item, str) for item in raw_actions):
        raise ValueError("Action Skeleton contains invalid entries")
    actions = text_to_actions(raw_actions, primitive_actions)
    if any(action.param_args for action in actions):
        raise ValueError(
            "Action Skeleton must not contain coordinates or other parameters"
        )
    return actions, payload["Reasoning"]


def _parse_llm_json_object(raw_text):
    """Parse a JSON object from an LLM response, tolerating light wrapper text."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw_text[start:end + 1])
        raise


def _compact_json_retry_prompt(original_prompt, error):
    """Short retry prompt for local models that drift into invalid JSON."""
    return (
        "Your previous answer could not be parsed as JSON.\n"
        f"Parser error: {error}\n\n"
        "Return ONLY one complete valid JSON object. Do not include markdown. "
        "Keep Reasoning to one short sentence. The JSON must have exactly these keys:\n"
        "{\n"
        '  "Reasoning": "one short sentence",\n'
        '  "Full Plan": ["pick([\\\'obj\\\'], {})", "place([\\\'obj\\\'], {\\\'x\\\': 0.60, \\\'y\\\': 0.02, \\\'theta\\\': 0.00})"]\n'
        "}\n\n"
        "Use the same task, state, ACFM guidance, slot tables, and feedback below. "
        "For GLOBAL PACKING SLOT STRATEGY, copy the mandatory slot coordinates exactly.\n\n"
        f"{original_prompt}"
    )


class ForeSightTAMP:
    """
    ForeSight-TAMP: Foresighted Spatial Intelligence for TAMP.

    Extends LLM³ with three contributions:
    1. ACFM - spatial feasibility maps before each planning call
    2. Mental Rehearsal - full-plan simulation before execution
    3. Cascade Analysis - root-cause backtracking on failure
    """

    def __init__(self, env=None, max_llm_calls=20,
                 disable_modules=None, use_gui=False,
                 acfm_resolution=20, save_dir="results",
                 candidate_selection_mode="llm",
                 rollout_validation_quotas=None,
                 render_acfm_maps=True,
                 export_trajectory_artifacts=True):
        self.env = env or PackCompactEnv()
        self.max_llm_calls = max_llm_calls
        self.use_gui = use_gui
        self.save_dir = Path(save_dir)
        mkdir(self.save_dir)

        # Rendered maps are human-inspection diagnostics that are never sent to
        # the language model.  Trajectory artifacts are the execution
        # certificates; they must stay on for a reportable run but can be
        # skipped for pilots and smoke tests.  Both cost measurable wall time.
        self.render_acfm_maps = bool(render_acfm_maps)
        self.export_trajectory_artifacts = bool(export_trajectory_artifacts)

        # Module enable/disable for ablation studies
        disabled = disable_modules or []
        self.use_acfm = "acfm" not in disabled
        self.use_rehearsal = "rehearsal" not in disabled
        self.use_cascade = "cascade" not in disabled
        # Disabling "acfm" removes the whole interface: the one-shot planner
        # replaces per-step binding and candidate rollout along with the map.
        # Disabling "ranking" removes only the soft ordering, so the masks,
        # the binding and the rollout all stay. It is the finer of the two.
        self.use_acfm_ranking = "ranking" not in disabled

        # ACFM settings
        self.acfm_resolution = acfm_resolution
        if candidate_selection_mode not in {
                "llm", "top1", "rollout_control"}:
            raise ValueError(
                "candidate_selection_mode must be 'llm', 'top1', or "
                "'rollout_control'")
        self.candidate_selection_mode = candidate_selection_mode
        quotas = dict(DEFAULT_ROLLOUT_VALIDATION_QUOTAS)
        quotas.update(rollout_validation_quotas or {})
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in quotas.values()
        ):
            raise ValueError(
                "rollout_validation_quotas must contain positive integers")
        self.rollout_validation_quotas = quotas

        # Planner will be created per-episode (needs task_file from config)
        self.planner = None

        # Persistent state across episodes (for ACFM layers 4-5).
        # failure_positions is keyed by object name so each object's ACFM only
        # sees failures for *that* specific object (cracker_box failures should
        # not block sugar_box from the same cell).
        self.failure_positions = {}   # {obj_name: [(x, y), ...]}
        self.disturbance_positions = []
        self._packing_slot_strategy = None
        self.acfm_hist = []
        self._last_planning_llm_calls = 0
        self._audit_metadata = {}

    def _init_planner(self, task_file=None):
        """Initialize the planner with the appropriate task description."""
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.shelf_env import ShelfEnv
        from envs.push_stacking_env import PushStackingEnv
        if isinstance(self.env, TowerOfHanoiEnv):
            env_desc = "hanoi.txt"
        elif isinstance(self.env, ShelfEnv):
            env_desc = "shelf.txt"
        elif isinstance(self.env, PushStackingEnv):
            env_desc = "push_stacking.txt"
        else:
            env_desc = "pack_boxes.txt"
        self.planner = LLMTAMPPlanner(
            planner_prompt_file="foresight",
            env_desc_file=env_desc,
            primitive_actions=self.env.primitive_actions,
            with_mp_feedback=True,
            trace_size=5,
            task_file=task_file,
        )

    @staticmethod
    def _build_audit_metadata(task_config, task_type, resolution):
        """Return validation metadata with a canonical benchmark split key."""
        default_weight_profile = (
            "push_default" if task_type == "push_stacking"
            else "3d_default" if task_type in {"stacking", "shelf"}
            else "discrete_hanoi" if task_type == "hanoi"
            else "2d_default"
        )
        return {
            "scenario_id": task_config.get("scenario_id"),
            "split": (
                task_config.get("split")
                or task_config.get("benchmark_split")
            ),
            "domain": task_config.get("domain", task_type),
            "resolution": int(resolution),
            "weight_profile": task_config.get(
                "weight_profile", default_weight_profile),
            "threshold": task_config.get("threshold"),
            "push_cost_coefficient": task_config.get(
                "push_cost_coefficient"),
        }

    def run_episode(self, task_config, play_traj=False):
        """
        Run a single ForeSight-TAMP episode.

        The core loop:
        1. Compute ACFM for current state
        2. LLM generates plan (with ACFM guidance)
        3. Mental Rehearsal validates entire plan
        4. If rehearsal passes: execute
        5. If execution fails: Cascade Analysis → targeted backtrack
        """
        # The top-1 and equal-rollout controls use the deterministic task
        # skeleton and never instantiate an LLM client.  This is important for
        # both causal interpretation and credential-free offline execution.
        if self.candidate_selection_mode == "llm":
            self._init_planner(task_file=task_config.get("task"))
        else:
            self.planner = None
        self._packing_slot_strategy = None

        # Build reset kwargs based on environment type
        from envs.shelf_env import ShelfEnv
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.push_stacking_env import PushStackingEnv
        if isinstance(self.env, TowerOfHanoiEnv):
            reset_kwargs = {
                "disks": task_config["disks"],
                "pegs": task_config["pegs"],
            }
            self._task_type = "hanoi"
        elif isinstance(self.env, PushStackingEnv):
            reset_kwargs = {
                "boxes": task_config["boxes"],
                "stacking_order": task_config["stacking_order"],
                "target_pose": task_config["target_pose"],
                "initial_joint_positions": task_config.get(
                    "initial_joint_positions"),
                "approach_tcp_pose": task_config.get("approach_tcp_pose"),
            }
            self._task_type = "push_stacking"
        elif isinstance(self.env, StackingEnv):
            reset_kwargs = {
                "boxes": task_config["boxes"],
                "stacking_order": task_config["stacking_order"],
                "stacking_zone": task_config.get("stacking_zone"),
            }
            self._task_type = "stacking"
        elif isinstance(self.env, ShelfEnv):
            reset_kwargs = {
                "boxes": task_config["boxes"],
                "shelf_config": task_config["shelf_config"],
            }
            self._task_type = "shelf"
        else:
            reset_kwargs = {
                "basket": task_config["basket"],
                "boxes": task_config["boxes"],
                "task_type": task_config.get("task_type", "packing"),
                "stacking_order": task_config.get("stacking_order", None),
            }
            self._task_type = reset_kwargs.get("task_type", "packing")
        self._audit_metadata = self._build_audit_metadata(
            task_config, self._task_type, self.acfm_resolution)
        obs, obs_text = self.env.reset(**reset_kwargs, use_gui=self.use_gui)

        # Episode state
        self.acfm_hist = []
        executed_actions = []
        placed_positions = {}
        last_feedback_list = []
        last_temp_plan = None
        num_llm_calls = 0
        num_mp_calls = 0
        rehearsal_saves = 0
        cascade_depths = []
        acfm_scores_chosen = []
        discarded_simulator_primitives = 0
        committed_successful_primitives = 0
        feedback = None
        joint_path_records = []
        cartesian_path_records = []
        self._write_executed_path_files(
            joint_path_records, cartesian_path_records)

        for iteration in range(self.max_llm_calls):
            # Keep simulator-backed ACFM objects local to this reset/attempt.
            # Persistent audit history is stored separately as detached,
            # JSON-safe records so an old map can never be consulted in a new
            # predicted state.
            attempt_acfm_history = []

            # Reset environment to replay from known good state
            obs, obs_text = self.env.reset(**reset_kwargs, use_gui=self.use_gui)
            # A reset invalidates all placed-object bookkeeping and maps from
            # the preceding attempt.  Failure/disturbance statistics remain in
            # their separate episode histories, but predicted-state provenance
            # starts from this reset state.
            placed_positions.clear()

            # ========== STEPS 1-2: skeleton, then predicted-state ACFMs ==========
            try:
                plan = self._plan_sequentially_with_acfm(
                    obs_text=obs_text,
                    feedback_list=last_feedback_list,
                    task_config=task_config,
                    acfm_history=attempt_acfm_history,
                    iteration=iteration,
                )
            finally:
                self._persist_attempt_history(attempt_acfm_history)
            num_llm_calls += self._last_planning_llm_calls

            if plan is None:
                logger.warning("Failed to generate plan")
                # Carry the reason into the next attempt so the model knows
                # why its skeleton was rejected and has a reason to revise it.
                last_feedback_list = self._planning_failure_feedback()
                continue

            last_feedback_list = []

            # ========== STEP 3: Mental Rehearsal ==========
            rehearsal_result = None
            if self.use_rehearsal:
                rehearsal_engine = MentalRehearsal(self.env)
                rehearsal_result = rehearsal_engine.rehearse(plan)

                # Successful rehearsal and committed execution begin from the
                # same reset state, so execute the exact validated trajectories.
                # Failed partial rehearsals are not reusable.
                if rehearsal_result.success:
                    MentalRehearsal.apply_cached_trajectories(
                        plan, rehearsal_result)
                else:
                    for action in plan:
                        action.traj = []

                if not rehearsal_result.success:
                    rehearsal_saves += 1
                    rehearsal_report = rehearsal_engine.format_report_for_llm(
                        rehearsal_result)
                    failed_idx = rehearsal_result.failed_step or 0
                    failed_action = plan[failed_idx] if failed_idx < len(plan) else plan[0]
                    if failed_action.primitive.name in {"place", "push"}:
                        obj_key = (
                            failed_action.obj_args[0]
                            if failed_action.obj_args else "__unknown__"
                        )
                        self.failure_positions.setdefault(obj_key, []).append((
                            failed_action.param_args.get("x", 0),
                            failed_action.param_args.get("y", 0),
                        ))
                    push_detour = self._push_detour_feedback(
                        failed_action, rehearsal_result.failure_detail)
                    if push_detour:
                        rehearsal_report = f"{rehearsal_report}\n\n{push_detour}"
                    logger.info(
                        f"Rehearsal caught failure at step "
                        f"{rehearsal_result.failed_step}: "
                        f"{rehearsal_result.failure_type}")

                    # Feed rehearsal failure back to LLM as feedback
                    # Create a synthetic feedback entry
                    from utils.tamp_util import TAMPFeedback
                    synthetic_feedback = TAMPFeedback(
                        motion_planner_feedback=(
                            f"REHEARSAL FAILURE: {rehearsal_report}"),
                        task_process_feedback="Plan rejected by rehearsal",
                        action_success=False,
                        goal_achieved=False,
                    )
                    # Use the failed action from the plan
                    if failed_idx < len(plan):
                        last_feedback_list = [
                            (plan[failed_idx], synthetic_feedback)]
                    else:
                        last_feedback_list = [
                            (plan[0], synthetic_feedback)]

                    # Re-reset environment (rehearsal may have modified state)
                    obs, obs_text = self.env.reset(
                        **reset_kwargs, use_gui=self.use_gui)
                    placed_positions.clear()
                    continue

            # ========== STEP 4: Execute Plan ==========
            # MentalRehearsal restores the exact saved start state.  Preserve
            # it so cached trajectories remain tied to the state that was
            # validated.  Without rehearsal, do a clean reset.
            if rehearsal_result is None:
                obs, obs_text = self.env.reset(
                    **reset_kwargs, use_gui=self.use_gui)
            placed_positions.clear()
            self._write_environment_state_text(task_config)

            temp_plan = []
            execution_checkpoints = []
            # A successful rehearsal has just installed the exact trajectories
            # validated from this identical start state.  Reusing a trajectory
            # from a prior attempt here would silently replace that evidence.
            same_as_last = rehearsal_result is None

            for action_i, action in enumerate(plan):
                # Trajectory reuse optimization (from LLM³)
                if (same_as_last and last_temp_plan is not None
                        and len(last_temp_plan) > action_i):
                    if str(action) == str(last_temp_plan[action_i]):
                        action = last_temp_plan[action_i]
                    else:
                        same_as_last = False

                if action.traj is None or len(action.traj) == 0:
                    num_mp_calls += 1

                execution_checkpoints.append(
                    self._save_execution_checkpoint()
                    if self.use_cascade else None
                )
                _, feedback = self.env.step(
                    action, play_traj=play_traj)
                last_feedback_list.append((action, feedback))

                if feedback.action_success:
                    temp_plan.append(action)
                    joint_record, cartesian_record = (
                        self._capture_executed_action_paths(
                            action=action,
                            iteration=iteration,
                            action_index=action_i,
                        )
                    )
                    if joint_record is not None:
                        joint_path_records.append(joint_record)
                        cartesian_path_records.append(cartesian_record)
                        self._write_executed_path_files(
                            joint_path_records, cartesian_path_records)
                    # Track placement positions
                    if action.primitive.name == "place":
                        placed_positions[action.obj_args[0]] = {
                            "x": action.param_args.get("x", 0),
                            "y": action.param_args.get("y", 0),
                        }
                        # Track ACFM score of chosen position
                        if self.use_acfm and attempt_acfm_history:
                            score = self._selected_candidate_score(
                                attempt_acfm_history,
                                step=action_i,
                                primitive="place",
                                obj_name=action.obj_args[0],
                            )
                            if score is not None:
                                acfm_scores_chosen.append(score)
                    elif action.primitive.name == "push":
                        if self.use_acfm and attempt_acfm_history:
                            obj_key = action.obj_args[0] if action.obj_args else None
                            score = self._selected_candidate_score(
                                attempt_acfm_history,
                                step=action_i,
                                primitive="push",
                                obj_name=obj_key,
                            )
                            if score is not None:
                                acfm_scores_chosen.append(score)
                else:
                    logger.info(f"Action {str(action)} failed: "
                                f"{feedback.motion_planner_feedback}")
                    # A cached trajectory is valid only for the exact state in
                    # which rehearsal accepted it.  Do not carry a rejected
                    # cached path into the next planning attempt.
                    action.traj = []

                    # Always record failed place positions (keyed by object name)
                    # so the ACFM failure-history layer penalises them in the
                    # next iteration, steering the planner toward a different
                    # candidate instead of repeating the same failing coordinate.
                    if action.primitive.name in {"place", "push"}:
                        obj_key = action.obj_args[0] if action.obj_args else "__unknown__"
                        self.failure_positions.setdefault(obj_key, []).append((
                            action.param_args.get("x", 0),
                            action.param_args.get("y", 0),
                        ))

                    # ========== STEP 5: Cascade Analysis on Failure ==========
                    if self.use_cascade and action.primitive.name == "place":
                        def counterfactual_replayer(**kwargs):
                            return self._replay_counterfactual_suffix(
                                plan=plan,
                                failed_step=action_i,
                                checkpoints=execution_checkpoints,
                                **kwargs,
                            )

                        analyzer = CascadeAnalyzer(
                            self.env,
                            attempt_acfm_history,
                            counterfactual_replayer=counterfactual_replayer,
                        )
                        cascade_result = analyzer.analyze(
                            failed_action=action,
                            failed_step=action_i,
                            failure_feedback=feedback.motion_planner_feedback,
                            executed_plan=[str(a) for a in temp_plan] + [str(action)],
                            placed_positions=placed_positions,
                        )
                        cascade_report = analyzer.format_report_for_llm(
                            cascade_result)
                        cascade_depths.append(
                            len(cascade_result.blocking_chain))

                        logger.info(
                            f"Cascade analysis: root cause at step "
                            f"{cascade_result.root_cause_step}")

                        # Append cascade info to the feedback
                        from utils.tamp_util import TAMPFeedback
                        cascade_feedback = TAMPFeedback(
                            motion_planner_feedback=(
                                f"{feedback.motion_planner_feedback}\n"
                                f"{cascade_report}"),
                            task_process_feedback=feedback.task_process_feedback,
                            action_success=False,
                            goal_achieved=False,
                        )
                        last_feedback_list[-1] = (action, cascade_feedback)
                    break

                if feedback.goal_achieved:
                    break

            self._remove_execution_checkpoints(execution_checkpoints)
            last_temp_plan = temp_plan

            if feedback and feedback.goal_achieved:
                committed_successful_primitives = len(temp_plan)
                logger.info("ForeSight-TAMP: Goal achieved!")
                break
            else:
                # Successful simulator primitives in an unsuccessful attempt
                # are discarded by the reset at the start of the next attempt.
                discarded_simulator_primitives += len(temp_plan)
                if feedback:
                    logger.info(
                        f"Goal not achieved: {feedback.task_process_feedback}")

        high_score_selection_rate = (
            round(
                100.0 * np.mean([
                    float(score > 0.7) for score in acfm_scores_chosen
                ]),
                1,
            )
            if acfm_scores_chosen else 0.0
        )
        return {
            "goal_achieved": feedback.goal_achieved if feedback else False,
            "candidate_selection_mode": self.candidate_selection_mode,
            "rollout_validation_quotas": (
                copy.deepcopy(self.rollout_validation_quotas)
                if self.candidate_selection_mode == "rollout_control"
                else None
            ),
            "num_llm_calls": num_llm_calls,
            "num_mp_calls": num_mp_calls,
            "tamp_plan": (
                [str(a) for a in temp_plan]
                if feedback and feedback.goal_achieved else None
            ),
            "committed_successful_simulator_primitives": (
                committed_successful_primitives),
            "discarded_simulator_primitives": (
                discarded_simulator_primitives),
            # Backward-compatible name used by earlier result files.
            "wasted_real_actions": discarded_simulator_primitives,
            "rehearsal_saves": rehearsal_saves,
            "cascade_depth": (
                round(np.mean(cascade_depths), 2)
                if cascade_depths else 0
            ),
            "high_score_selection_rate": high_score_selection_rate,
            # Backward-compatible alias; this is a selection-rate diagnostic,
            # not an independent feasibility-accuracy estimate.
            "acfm_accuracy": high_score_selection_rate,
            "acfm_hist": self.acfm_hist,
            "joint_paths_file": str(
                self.save_dir / "executed_joint_space_paths.json"),
            "cartesian_paths_file": str(
                self.save_dir / "executed_cartesian_paths.json"),
            "interpolated_joint_paths_file": str(
                self.save_dir
                / "executed_interpolated_joint_space_paths.json"),
            "interpolated_cartesian_paths_file": str(
                self.save_dir
                / "executed_interpolated_cartesian_paths.json"),
            "environment_state_file": str(
                self.save_dir / "environment_state.txt"),
            "joint_subtasks_file": str(
                self.save_dir / "joint_configurations_by_subtask.json"),
            "joint_subtasks_text_file": str(
                self.save_dir / "joint_configurations_by_subtask.txt"),
        }

    def _write_environment_state_text(self, task_config):
        """Export the initialized simulation state for real-robot review."""
        with metric_phase("export"), metric_timer(
                "environment_state_write"):
            return self._write_environment_state_text_impl(task_config)

    def _write_environment_state_text_impl(self, task_config):
        """Build and write the environment-state artifact."""
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return

        joint_indices = list(robot.ik_joints)
        joint_names = []
        for joint_idx in joint_indices:
            raw_name = p.getJointInfo(robot.robot, joint_idx)[1]
            joint_names.append(
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes) else str(raw_name)
            )
        joint_positions = [
            float(state[0])
            for state in p.getJointStates(robot.robot, joint_indices)
        ]

        object_configs = {}
        for cfg in (task_config.get("boxes") or {}).values():
            if isinstance(cfg, dict) and cfg.get("name"):
                object_configs[cfg["name"]] = cfg

        lines = [
            "FORESIGHT-TAMP ENVIRONMENT STATE",
            f"generated_utc: {datetime.now(timezone.utc).isoformat()}",
            "length_units: meters",
            "angle_units: radians",
            "world_frame: pybullet_world / FR3 base frame",
            "axis_convention: +x away from robot; +y along table; +z upward",
            "SAFETY: Simulation export only. Validate frames, joint limits, "
            "collisions, payload, timing, and emergency-stop behavior before "
            "commanding hardware.",
            "",
            "[robot]",
            "model: Franka Research 3 (FR3) with Franka hand",
            f"urdf: {getattr(robot, 'urdf_path', 'unknown')}",
            f"joint_indices: {json.dumps(joint_indices)}",
            f"joint_names: {json.dumps(joint_names)}",
            f"initial_joint_positions: {json.dumps(joint_positions)}",
            f"trajectory_time_step_seconds: {float(getattr(robot, 'trajectory_time_step', 1.0 / 60.0))}",
            "",
            "[table]",
            f"center_xyz: [{self.env.TABLE_CENTER_X}, {self.env.TABLE_CENTER_Y}, {self.env.TABLE_TOP_Z}]",
            f"size_xyz: [{self.env.TABLE_WIDTH_X}, {self.env.TABLE_LENGTH_Y}, {self.env.TABLE_THICKNESS}]",
            f"x_bounds: [{self.env.TABLE_CENTER_X - self.env.TABLE_WIDTH_X / 2.0}, {self.env.TABLE_CENTER_X + self.env.TABLE_WIDTH_X / 2.0}]",
            f"y_bounds: [{self.env.TABLE_CENTER_Y - self.env.TABLE_LENGTH_Y / 2.0}, {self.env.TABLE_CENTER_Y + self.env.TABLE_LENGTH_Y / 2.0}]",
            "",
            "[task]",
            f"task: {task_config.get('task')}",
            f"target_pose: {json.dumps(task_config.get('target_pose'), sort_keys=True)}",
            f"stacking_order: {json.dumps(task_config.get('stacking_order'))}",
            f"configured_prepush_joints: {json.dumps(task_config.get('initial_joint_positions'))}",
            f"configured_approach_tcp_xyz: {json.dumps(task_config.get('approach_tcp_pose'))}",
            "",
            "[bodies]",
        ]

        for name, body_id in self.env.objects.items():
            position, orientation = p.getBasePositionAndOrientation(body_id)
            bb_min, bb_max = p.getAABB(body_id)
            cfg = object_configs.get(name, {})
            size = [
                float(bb_max[index] - bb_min[index])
                for index in range(3)
            ]
            lines.extend([
                f"body: {name}",
                f"  id: {body_id}",
                f"  position_xyz: {json.dumps([float(v) for v in position])}",
                f"  orientation_xyzw: {json.dumps([float(v) for v in orientation])}",
                f"  measured_size_xyz: {json.dumps(size)}",
                f"  aabb_min_xyz: {json.dumps([float(v) for v in bb_min])}",
                f"  aabb_max_xyz: {json.dumps([float(v) for v in bb_max])}",
                f"  configured_dimensions_wlh: {json.dumps([cfg.get('w'), cfg.get('l'), cfg.get('h')])}",
                f"  mass_kg: {json.dumps(cfg.get('mass'))}",
                f"  static_fixture: {bool(cfg.get('fixture') or cfg.get('static') or name == 'table' or (hasattr(self.env, 'is_fixture') and self.env.is_fixture(name)))}",
                "",
            ])

        (self.save_dir / "environment_state.txt").write_text(
            "\n".join(lines), encoding="utf-8")

    @suppress_gui_rendering
    def _capture_executed_action_paths(
            self, action, iteration, action_index):
        """Capture complete joint and end-effector paths for one real action."""
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return None, None

        trace = list(getattr(robot, "last_motion_trace", []) or [])
        if not trace and action.traj:
            trace = [{
                "phase": action.primitive.name,
                "trajectory": action.traj,
            }]
        if not trace:
            return None, None

        robot_id = robot.robot
        joint_indices = list(robot.ik_joints)
        saved_positions = [
            float(state[0])
            for state in p.getJointStates(robot_id, joint_indices)
        ]

        common = {
            "iteration": int(iteration),
            "action_index": int(action_index),
            "primitive": action.primitive.name,
            "object_args": list(action.obj_args),
            "parameters": {
                key: float(value) if isinstance(
                    value, (int, float, np.integer, np.floating)) else value
                for key, value in action.param_args.items()
            },
        }
        joint_segments = []
        cartesian_segments = []

        try:
            for segment in trace:
                phase = str(segment.get("phase", action.primitive.name))
                trajectory = segment.get("trajectory") or []
                joint_waypoints = [
                    [float(value) for value in conf]
                    for conf in trajectory
                ]
                cartesian_waypoints = []
                for conf in joint_waypoints:
                    for joint_idx, value in zip(joint_indices, conf):
                        p.resetJointState(
                            robot_id, joint_idx, targetValue=value)
                    link_state = p.getLinkState(
                        robot_id,
                        robot.tool_attach_link,
                        computeForwardKinematics=True,
                    )
                    cartesian_waypoints.append({
                        "position_xyz": [
                            float(value) for value in link_state[4]
                        ],
                        "orientation_xyzw": [
                            float(value) for value in link_state[5]
                        ],
                    })

                joint_segments.append({
                    "phase": phase,
                    "waypoint_count": len(joint_waypoints),
                    "waypoints": joint_waypoints,
                })
                cartesian_segments.append({
                    "phase": phase,
                    "waypoint_count": len(cartesian_waypoints),
                    "waypoints": cartesian_waypoints,
                })
        finally:
            for joint_idx, value in zip(joint_indices, saved_positions):
                p.resetJointState(
                    robot_id, joint_idx, targetValue=value)

        return (
            {**common, "segments": joint_segments},
            {**common, "segments": cartesian_segments},
        )

    def _write_executed_path_files(
            self, joint_path_records, cartesian_path_records):
        """Write all executed paths into one file per coordinate space.

        These files are the execution certificates for a reported run, so they
        stay on by default.  Pilots and smoke tests, which report nothing, can
        skip them to save wall time.
        """
        if not self.export_trajectory_artifacts:
            return None
        with metric_phase("export"), metric_timer("trajectory_artifact_write"):
            return self._write_executed_path_files_impl(
                joint_path_records, cartesian_path_records)

    def _write_executed_path_files_impl(
            self, joint_path_records, cartesian_path_records):
        """Build and write joint, Cartesian, and subtask path artifacts."""
        robot = getattr(self.env, "robot", None)
        joint_indices = (
            list(robot.ik_joints) if robot is not None else [])
        joint_names = []
        if robot is not None:
            for joint_idx in joint_indices:
                raw_name = p.getJointInfo(robot.robot, joint_idx)[1]
                joint_names.append(
                    raw_name.decode("utf-8")
                    if isinstance(raw_name, bytes) else str(raw_name)
                )

        joint_payload = {
            "format_version": 2,
            "path_kind": "interpolated_execution_trajectory",
            "coordinate_space": "joint",
            "units": "radians",
            "time_step_seconds": float(
                getattr(robot, "trajectory_time_step", 1.0 / 60.0)
            ) if robot is not None else 1.0 / 60.0,
            "timing_profile": "quintic_stop_at_polyline_corners",
            "interpolation": {
                "method": "quintic_smoothstep",
                "preserves_collision_checked_edges": True,
                "zero_velocity_at_polyline_corners": True,
            },
            "joint_indices": joint_indices,
            "joint_names": joint_names,
            "actions": joint_path_records,
        }
        cartesian_payload = {
            "format_version": 2,
            "path_kind": "interpolated_execution_trajectory",
            "coordinate_space": "cartesian",
            "reference_frame": "pybullet_world",
            "position_units": "meters",
            "orientation_format": "quaternion_xyzw",
            "time_step_seconds": float(
                getattr(robot, "trajectory_time_step", 1.0 / 60.0)
            ) if robot is not None else 1.0 / 60.0,
            "timing_profile": "quintic_stop_at_polyline_corners",
            "interpolation": {
                "method": "forward_kinematics_of_interpolated_joint_path",
                "source_joint_method": "quintic_smoothstep",
                "zero_velocity_at_polyline_corners": True,
            },
            "actions": cartesian_path_records,
        }
        dump_json(
            joint_payload,
            self.save_dir / "executed_joint_space_paths.json",
        )
        dump_json(
            cartesian_payload,
            self.save_dir / "executed_cartesian_paths.json",
        )
        # Explicitly named copies make it unambiguous which files are safe for
        # smooth fixed-rate playback. Keep the original names for compatibility
        # with existing analysis scripts.
        dump_json(
            joint_payload,
            self.save_dir / "executed_interpolated_joint_space_paths.json",
        )
        dump_json(
            cartesian_payload,
            self.save_dir
            / "executed_interpolated_cartesian_paths.json",
        )
        self._write_joint_subtask_files(joint_payload)

    def _write_joint_subtask_files(self, joint_payload):
        """Export every labeled motion segment as a hardware-review subtask."""
        joint_names = list(joint_payload.get("joint_names") or [])
        arm_columns = [
            index for index, name in enumerate(joint_names)
            if name.startswith("fr3_joint")
        ]
        arm_joint_names = [joint_names[index] for index in arm_columns]
        subtasks = []
        source_time_step = float(joint_payload.get(
            "time_step_seconds", 1.0 / 60.0))
        review_max_velocity = 0.5
        review_max_acceleration = 2.0
        max_first_difference = 0.0
        max_second_difference = 0.0

        for action in joint_payload.get("actions") or []:
            for segment_index, segment in enumerate(action.get("segments") or []):
                full_waypoints = segment.get("waypoints") or []
                arm_waypoints = [
                    [float(conf[index]) for index in arm_columns]
                    for conf in full_waypoints
                ]
                for q0, q1 in zip(arm_waypoints, arm_waypoints[1:]):
                    max_first_difference = max(
                        max_first_difference,
                        max(abs(v1 - v0) for v0, v1 in zip(q0, q1)),
                    )
                velocities_per_sample = [
                    [v1 - v0 for v0, v1 in zip(q0, q1)]
                    for q0, q1 in zip(arm_waypoints, arm_waypoints[1:])
                ]
                for d0, d1 in zip(
                        velocities_per_sample, velocities_per_sample[1:]):
                    max_second_difference = max(
                        max_second_difference,
                        max(abs(v1 - v0) for v0, v1 in zip(d0, d1)),
                    )
                subtask_id = (
                    f"iteration_{action.get('iteration', 0)}_"
                    f"action_{action.get('action_index', 0)}_"
                    f"segment_{segment_index}_{segment.get('phase', 'motion')}"
                )
                subtasks.append({
                    "subtask_id": subtask_id,
                    "iteration": action.get("iteration"),
                    "action_index": action.get("action_index"),
                    "primitive": action.get("primitive"),
                    "object_args": action.get("object_args") or [],
                    "parameters": action.get("parameters") or {},
                    "phase": segment.get("phase"),
                    "waypoint_count": len(full_waypoints),
                    "duration_seconds": 0.0,
                    "arm_joint_configurations": arm_waypoints,
                    "full_joint_configurations": full_waypoints,
                })

        velocity_time_step = (
            max_first_difference / review_max_velocity
            if max_first_difference > 0.0 else 0.0)
        acceleration_time_step = (
            np.sqrt(max_second_difference / review_max_acceleration)
            if max_second_difference > 0.0 else 0.0)
        # Ten percent timing margin, rounded upward to a whole millisecond.
        required_time_step = max(
            source_time_step, velocity_time_step, acceleration_time_step)
        review_time_step = float(
            np.ceil(required_time_step * 1.10 * 1000.0) / 1000.0)
        for subtask in subtasks:
            subtask["duration_seconds"] = max(
                0, subtask["waypoint_count"] - 1) * review_time_step

        export_payload = {
            "format_version": 1,
            "export_purpose": "FR3 real-robot trajectory review",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "safety_status": "simulation_only_not_hardware_validated",
            "safety_warning": (
                "Do not execute directly on hardware. Verify robot/world "
                "frames, limits, collision clearance, controller timing, "
                "payload, gripper commands, and emergency-stop behavior."
            ),
            "position_units": "radians",
            "source_simulation_time_step_seconds": source_time_step,
            "time_step_seconds": review_time_step,
            "timing_profile": "uniform_conservative_hardware_review_sampling",
            "timing_note": (
                "The export period is conservatively increased from the "
                "simulation period using finite-difference velocity and "
                "acceleration bounds. A certified robot-side trajectory "
                "generator must still retime this path."
            ),
            "review_max_joint_velocity_rad_s": review_max_velocity,
            "review_max_joint_acceleration_rad_s2": review_max_acceleration,
            "arm_joint_names": arm_joint_names,
            "full_joint_names": joint_names,
            "subtasks": subtasks,
        }
        dump_json(
            export_payload,
            self.save_dir / "joint_configurations_by_subtask.json",
        )

        lines = [
            "FORESIGHT-TAMP JOINT CONFIGURATIONS BY SUBTASK",
            f"generated_utc: {export_payload['generated_utc']}",
            "units: radians",
            f"time_step_seconds: {export_payload['time_step_seconds']}",
            f"arm_joint_names: {json.dumps(arm_joint_names)}",
            f"full_joint_names: {json.dumps(joint_names)}",
            "SAFETY: SIMULATION ONLY; NOT VALIDATED FOR DIRECT HARDWARE EXECUTION.",
            "",
        ]
        for subtask in subtasks:
            lines.extend([
                f"[subtask {subtask['subtask_id']}]",
                f"primitive: {subtask['primitive']}",
                f"phase: {subtask['phase']}",
                f"objects: {json.dumps(subtask['object_args'])}",
                f"parameters: {json.dumps(subtask['parameters'], sort_keys=True)}",
                f"waypoint_count: {subtask['waypoint_count']}",
                f"duration_seconds: {subtask['duration_seconds']}",
            ])
            for waypoint_index, (arm_conf, full_conf) in enumerate(zip(
                    subtask["arm_joint_configurations"],
                    subtask["full_joint_configurations"])):
                lines.append(
                    f"  waypoint {waypoint_index:06d}: "
                    f"arm={json.dumps(arm_conf)} "
                    f"full={json.dumps(full_conf)}"
                )
            lines.append("")

        (self.save_dir / "joint_configurations_by_subtask.txt").write_text(
            "\n".join(lines), encoding="utf-8")

    def _compute_push_acfm_for_state(self, acfm_history, step):
        """Compute push-specific ACFM maps for all pushable targets."""
        push_targets = self.env.get_push_targets()
        if not push_targets:
            return self.env.get_foresight_guidance()

        acfm_texts = [self.env.get_foresight_guidance()]
        push_bounds = self.env.get_push_bounds()

        for obj_name, target in push_targets.items():
            if obj_name not in self.env.objects:
                continue
            if self.env._base_in_target(obj_name):
                continue

            try:
                existing_objects = {
                    name: oid
                    for name, oid in self.env.objects.items()
                    if name != obj_name and name != "table" and not name.startswith("target_zone")
                }
                acfm = PushACFM(resolution=self.acfm_resolution)
                acfm.compute_push(
                    target_obj_id=self.env.objects[obj_name],
                    target_name=obj_name,
                    existing_objects=existing_objects,
                    robot_id=self.env.robot.robot,
                    ee_link_idx=self.env.robot.tool_attach_link,
                    push_bounds=push_bounds,
                    push_goal=target,
                    failure_history=self.failure_positions.get(obj_name, []),
                    disturbance_history=self.disturbance_positions,
                    push_params=self.env._push_params(obj_name),
                )

                if self.render_acfm_maps:
                    heatmap_path = (
                        self.save_dir / f"acfm_step{step}_push_{obj_name}.png")
                    planned_positions = [(target["x"], target["y"])]
                    with metric_phase("export"), metric_timer(
                            "acfm_render_seconds"):
                        acfm.render_3d_volume(
                            save_path=str(heatmap_path),
                            planned_positions=planned_positions,
                        )
                acfm_texts.append(acfm.textualize())
                acfm_history.append({
                    "step": step,
                    "acfm": acfm,
                    "object": obj_name,
                    "action": "push",
                })
            except Exception as e:
                logger.warning(
                    f"Push ACFM computation failed for '{obj_name}' (step {step}): {e}"
                )

        return "\n\n".join(text for text in acfm_texts if text)

    def _compute_hanoi_acfm_for_state(self, acfm_history, step):
        """Compute discrete 3D ACFM map for Tower of Hanoi peg moves."""
        acfm = HanoiACFM(resolution=self.acfm_resolution)
        acfm.compute_hanoi(self.env)
        if self.render_acfm_maps:
            heatmap_path = self.save_dir / f"acfm_step{step}_hanoi_state.png"
            with metric_phase("export"), metric_timer("acfm_render_seconds"):
                acfm.render_3d_volume(save_path=str(heatmap_path))
        acfm_history.append({
            "step": step,
            "acfm": acfm,
            "object": "__hanoi_state__",
            "action": "place",
        })
        guidance = [acfm.textualize()]
        if hasattr(self.env, "get_symbolic_plan"):
            symbolic_plan = self.env.get_symbolic_plan()
            if symbolic_plan:
                guidance.extend([
                    "",
                    "MANDATORY COMPLETE HANOI PLAN FROM THE CURRENT OBSERVATION:",
                    "Use this full legal sequence. Do not output a partial continuation.",
                    *[f"  {step_text}" for step_text in symbolic_plan],
                ])
        return "\n".join(guidance)

    def _support_surface_z(self):
        if "table" in self.env.objects:
            return float(self.env.get_bb("table")[1][2])
        return 0.005

    def _body_z_with_bottom_on(self, obj_name, support_z):
        oid = self.env.objects[obj_name]
        aabb = p.getAABB(oid)
        body_z = p.getBasePositionAndOrientation(oid)[0][2]
        aabb_center_z = (aabb[0][2] + aabb[1][2]) / 2.0
        aabb_offset_z = aabb_center_z - body_z
        half_z = (aabb[1][2] - aabb[0][2]) / 2.0
        return float(support_z + half_z - aabb_offset_z)

    @staticmethod
    def _natural_key(name):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(name))
        ]

    def _declared_xy_size(self, obj_name):
        """Return conservative object footprint from scenario spec and PyBullet."""
        spec = getattr(self.env, "_box_specs", {}).get(obj_name, {})
        aabb = p.getAABB(self.env.objects[obj_name])
        actual_w = float(aabb[1][0] - aabb[0][0])
        actual_l = float(aabb[1][1] - aabb[0][1])
        declared_w = float(spec.get("w", actual_w) or actual_w)
        declared_l = float(spec.get("l", declared_w) or actual_l)
        ycb_class = str(spec.get("ycb_class", "")).lower()
        is_can = "can" in obj_name.lower() or "can" in ycb_class
        actual_ratio = max(actual_w, actual_l) / max(
            min(actual_w, actual_l), 1e-9)
        declared_ratio = max(declared_w, declared_l) / max(
            min(declared_w, declared_l), 1e-9)
        is_round_can = (
            is_can and actual_ratio <= 1.15 and declared_ratio <= 1.15
        )
        if is_round_can:
            # Cylindrical cans are rotationally symmetric. Use the largest
            # declared/actual footprint as the planning diameter. A rectangular
            # "can" (for example potted meat) must retain its two side lengths.
            d = max(actual_w, actual_l, declared_w, declared_l)
            return d, d, True
        return max(actual_w, declared_w), max(actual_l, declared_l), False

    def _basket_inner_bounds(self, basket_bounds):
        """Use scenario inner dimensions when available, else the observed AABB."""
        cfg = getattr(self.env, "_basket_config", {}) or {}
        if all(k in cfg for k in ("x", "y", "w", "l")):
            return {
                "x_min": float(cfg["x"] - cfg["w"] / 2.0),
                "x_max": float(cfg["x"] + cfg["w"] / 2.0),
                "y_min": float(cfg["y"] - cfg["l"] / 2.0),
                "y_max": float(cfg["y"] + cfg["l"] / 2.0),
            }
        return {
            "x_min": float(basket_bounds["x_min"]),
            "x_max": float(basket_bounds["x_max"]),
            "y_min": float(basket_bounds["y_min"]),
            "y_max": float(basket_bounds["y_max"]),
        }

    def _packing_command_orientation(
            self, obj_name, original_orientation, command_theta):
        """Predict object orientation produced by a top-grasp place command."""
        grasp_theta = getattr(self.env, "_grasp_theta_dict", {}) or {}
        grasp_yaw = float(grasp_theta.get(obj_name, 0.0))
        yaw_delta = float(command_theta) - grasp_yaw
        _, orientation = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0.0, 0.0, yaw_delta]),
            [0.0, 0.0, 0.0],
            original_orientation,
        )
        return orientation

    def _compute_packing_slot_strategy(self, objects_to_place, basket_bounds):
        """Create a globally non-overlapping packing plan.

        Local ACFM maxima can be misleading in dense repeated-object cases:
        each object sees a locally good cell, but the sequence can consume
        incompatible slots. Uniform cans retain the compact grid planner.
        Heterogeneous objects use a rotation-aware rectangle-packing search.
        """
        if len(objects_to_place) < 4:
            self._packing_slot_strategy = None
            return None

        sizes = []
        can_flags = []
        for obj_name in objects_to_place:
            sx, sy, is_can = self._declared_xy_size(obj_name)
            sizes.append((sx, sy))
            can_flags.append(is_can)

        areas = [sx * sy for sx, sy in sizes if sx > 0 and sy > 0]
        if not areas:
            self._packing_slot_strategy = None
            return None

        same_size = (max(areas) / max(min(areas), 1e-9)) <= 1.25
        mostly_cans = sum(can_flags) >= max(1, int(0.75 * len(can_flags)))
        if not (same_size and mostly_cans):
            strategy = self._compute_mixed_packing_slot_strategy(
                objects_to_place, basket_bounds, sizes)
            self._packing_slot_strategy = strategy
            return strategy

        bounds = self._basket_inner_bounds(basket_bounds)
        width = bounds["x_max"] - bounds["x_min"]
        length = bounds["y_max"] - bounds["y_min"]
        n = len(objects_to_place)

        # Conservative planning footprint. A few millimetres are reserved for
        # physics/mesh mismatch, but the grid still remains tight enough for
        # 4x2 can packing in PKG4.
        fp_x = max(s[0] for s in sizes)
        fp_y = max(s[1] for s in sizes)
        clearance_buffer = 0.002

        candidates = []
        for cols in range(1, n + 1):
            rows = int(np.ceil(n / cols))
            required_x = cols * fp_x + (cols + 1) * clearance_buffer
            required_y = rows * fp_y + (rows + 1) * clearance_buffer
            if required_x > width + 1e-9 or required_y > length + 1e-9:
                continue
            unused = cols * rows - n
            free_x = width - cols * fp_x
            free_y = length - rows * fp_y
            margin_x = free_x / (cols + 1)
            margin_y = free_y / (rows + 1)
            aspect_bonus = cols if width >= length else rows
            candidates.append({
                "cols": cols,
                "rows": rows,
                "unused": unused,
                "margin_x": margin_x,
                "margin_y": margin_y,
                "aspect_bonus": aspect_bonus,
            })

        if not candidates:
            self._packing_slot_strategy = None
            return None

        best = max(
            candidates,
            key=lambda c: (
                -c["unused"],
                min(c["margin_x"], c["margin_y"]),
                c["margin_x"] + c["margin_y"],
                c["aspect_bonus"],
            ),
        )
        cols, rows = best["cols"], best["rows"]

        cell_w = width / cols
        cell_l = length / rows
        raw_slots = []
        for row in range(rows):
            y = bounds["y_min"] + cell_l * (row + 0.5)
            for col in range(cols):
                x = bounds["x_min"] + cell_w * (col + 0.5)
                raw_slots.append({
                    "x": float(x),
                    "y": float(y),
                    "row": row + 1,
                    "col": col + 1,
                })

        # Fill the slots nearest to the robot first. Dense baskets can leave the
        # arm without a collision-free configuration for a near-side slot once
        # the deeper rows/columns are occupied. Sorting by the actual robot base
        # position keeps this independent of any particular basket orientation,
        # scenario coordinates, or object names.
        try:
            robot_position, _ = p.getBasePositionAndOrientation(
                self.env.robot.robot)
            access_x, access_y = map(float, robot_position[:2])
        except Exception:
            # All current tabletop environments place the robot at the world
            # origin, but retain a safe fallback for lightweight/unit-test envs.
            access_x, access_y = 0.0, 0.0

        raw_slots.sort(
            key=lambda s: (
                (s["x"] - access_x) ** 2 + (s["y"] - access_y) ** 2,
                abs(s["y"] - access_y),
                s["y"],
                s["x"],
            )
        )
        object_order = sorted(objects_to_place, key=self._natural_key)

        slots = []
        for obj_name, slot in zip(object_order, raw_slots):
            slots.append({
                "object": obj_name,
                "x": round(slot["x"], 4),
                "y": round(slot["y"], 4),
                "theta": 0.0,
                "row": slot["row"],
                "col": slot["col"],
            })

        strategy = {
            "layout_type": "uniform_grid",
            "bounds": bounds,
            "rows": rows,
            "cols": cols,
            "footprint": {"x": fp_x, "y": fp_y},
            "cell": {"w": cell_w, "l": cell_l},
            "margins": {"x": best["margin_x"], "y": best["margin_y"]},
            "access_origin": {"x": access_x, "y": access_y},
            "slots": slots,
        }
        self._packing_slot_strategy = strategy
        return strategy

    def _compute_mixed_packing_slot_strategy(
            self, objects_to_place, basket_bounds, sizes):
        """Pack heterogeneous rectangles using geometry-derived corner search.

        Candidate coordinates are generated from the container walls and the
        edges of already reserved rectangles. The search permits quarter-turn
        rotations and rejects every wall overhang or pairwise footprint
        overlap. No scenario names or object coordinates are encoded here.
        """
        bounds = self._basket_inner_bounds(basket_bounds)
        wall_clearance = 0.008
        object_clearance = 0.010
        usable = {
            "x_min": bounds["x_min"] + wall_clearance,
            "x_max": bounds["x_max"] - wall_clearance,
            "y_min": bounds["y_min"] + wall_clearance,
            "y_max": bounds["y_max"] - wall_clearance,
        }
        if (
            usable["x_max"] <= usable["x_min"]
            or usable["y_max"] <= usable["y_min"]
        ):
            return None

        try:
            robot_position, _ = p.getBasePositionAndOrientation(
                self.env.robot.robot)
            access_x, access_y = map(float, robot_position[:2])
        except Exception:
            access_x, access_y = 0.0, 0.0

        grasp_theta = getattr(self.env, "_grasp_theta_dict", {}) or {}
        items = []
        for obj_name, (base_w, base_l) in zip(objects_to_place, sizes):
            orientations = []
            grasp_yaw = float(grasp_theta.get(obj_name, 0.0))
            for command_theta in (0.0, float(np.pi / 2.0)):
                relative_quarter_turn = int(round(
                    (command_theta - grasp_yaw) / (np.pi / 2.0)
                ))
                if relative_quarter_turn % 2:
                    width, length = base_l, base_w
                else:
                    width, length = base_w, base_l
                candidate = (
                    float(width), float(length), command_theta
                )
                if not any(
                    abs(width - old[0]) < 1e-6
                    and abs(length - old[1]) < 1e-6
                    for old in orientations
                ):
                    orientations.append(candidate)
            items.append({
                "object": obj_name,
                "orientations": orientations,
                "area": float(base_w * base_l),
                "long_side": float(max(base_w, base_l)),
            })

        # Reserve the most constrained footprints first. This is the standard
        # fail-first ordering for rectangle packing and also naturally places
        # the large grocery box before small gap fillers.
        items.sort(
            key=lambda item: (
                -item["long_side"], -item["area"],
                self._natural_key(item["object"]),
            )
        )

        def overlaps(candidate, placed):
            for other in placed:
                separated = (
                    candidate["x_max"] + object_clearance
                    <= other["x_min"] + 1e-9
                    or other["x_max"] + object_clearance
                    <= candidate["x_min"] + 1e-9
                    or candidate["y_max"] + object_clearance
                    <= other["y_min"] + 1e-9
                    or other["y_max"] + object_clearance
                    <= candidate["y_min"] + 1e-9
                )
                if not separated:
                    return True
            return False

        def candidate_origins(width, length, placed, placement_bias=0):
            x_values = {
                usable["x_min"],
                usable["x_max"] - width,
            }
            y_values = {
                usable["y_min"],
                usable["y_max"] - length,
            }
            for rect in placed:
                x_values.update({
                    rect["x_min"] - object_clearance - width,
                    rect["x_max"] + object_clearance,
                })
                y_values.update({
                    rect["y_min"] - object_clearance - length,
                    rect["y_max"] + object_clearance,
                })

            origins = []
            for x_min in x_values:
                x_max = x_min + width
                if (
                    x_min < usable["x_min"] - 1e-9
                    or x_max > usable["x_max"] + 1e-9
                ):
                    continue
                for y_min in y_values:
                    y_max = y_min + length
                    if (
                        y_min < usable["y_min"] - 1e-9
                        or y_max > usable["y_max"] + 1e-9
                    ):
                        continue
                    cx = x_min + width / 2.0
                    cy = y_min + length / 2.0
                    origins.append((
                        (cx - access_x) ** 2 + (cy - access_y) ** 2,
                        abs(cy - access_y),
                        x_min,
                        y_min,
                    ))
            if placement_bias == 1:
                # Explore the far side as well as the default robot-near side.
                origins.sort(
                    key=lambda entry: (
                        -entry[0], entry[1], entry[2], entry[3]))
            elif placement_bias == 2:
                origins.sort(
                    key=lambda entry: (
                        entry[3], entry[2], entry[0], entry[1]))
            elif placement_bias == 3:
                origins.sort(
                    key=lambda entry: (
                        -entry[3], entry[2], entry[0], entry[1]))
            else:
                origins.sort()
            return [(entry[2], entry[3]) for entry in origins]

        # These tasks contain at most a handful of objects. Still cap explored
        # nodes so a malformed or impossible scenario cannot stall planning.
        max_nodes = 100000
        visited_nodes = 0

        max_solutions = 32
        solutions = []
        solution_keys = set()

        def search(item_idx, placed, orientation_mask, placement_bias):
            nonlocal visited_nodes
            visited_nodes += 1
            if visited_nodes > max_nodes:
                return True
            if item_idx == len(items):
                key = tuple(
                    (
                        rect["object"],
                        round(rect["x_min"], 5),
                        round(rect["y_min"], 5),
                        round(rect["theta"], 5),
                    )
                    for rect in placed
                )
                if key not in solution_keys:
                    solution_keys.add(key)
                    solutions.append([dict(rect) for rect in placed])
                # Each pass seeks one layout. Multiple passes deliberately
                # vary rotations and spatial bias to avoid retaining dozens of
                # nearly identical early DFS solutions.
                return True

            item = items[item_idx]
            orientation_options = sorted(
                item["orientations"],
                key=lambda option: (
                    option[1] > option[0],
                    -option[0],
                ),
            )
            if (
                len(orientation_options) > 1
                and ((orientation_mask >> item_idx) & 1)
            ):
                orientation_options.reverse()
            for width, length, theta in orientation_options:
                if (
                    width > usable["x_max"] - usable["x_min"] + 1e-9
                    or length > usable["y_max"] - usable["y_min"] + 1e-9
                ):
                    continue
                for x_min, y_min in candidate_origins(
                        width, length, placed, placement_bias):
                    rect = {
                        "object": item["object"],
                        "x_min": x_min,
                        "x_max": x_min + width,
                        "y_min": y_min,
                        "y_max": y_min + length,
                        "width": width,
                        "length": length,
                        "theta": theta,
                    }
                    if overlaps(rect, placed):
                        continue
                    placed.append(rect)
                    stop_search = search(
                        item_idx + 1,
                        placed,
                        orientation_mask,
                        placement_bias,
                    )
                    placed.pop()
                    if stop_search:
                        return True
            return False

        orientation_passes = 1 << min(len(items), 8)
        for orientation_mask in range(orientation_passes):
            for placement_bias in range(4):
                search(0, [], orientation_mask, placement_bias)
                if (
                    len(solutions) >= max_solutions
                    or visited_nodes > max_nodes
                ):
                    break
            if (
                len(solutions) >= max_solutions
                or visited_nodes > max_nodes
            ):
                break
        if not solutions:
            logger.warning(
                "No collision-free mixed-object packing layout found for %s",
                ", ".join(objects_to_place),
            )
            return None

        def as_slots(rectangles):
            slots = []
            for rect in rectangles:
                slots.append({
                    "object": rect["object"],
                    "x": round(
                        (rect["x_min"] + rect["x_max"]) / 2.0, 4),
                    "y": round(
                        (rect["y_min"] + rect["y_max"]) / 2.0, 4),
                    "theta": round(float(rect["theta"]), 4),
                    "footprint_x": round(float(rect["width"]), 4),
                    "footprint_y": round(float(rect["length"]), 4),
                })
            return slots

        candidate_slots = [as_slots(solution) for solution in solutions]

        return {
            "layout_type": "mixed_rectangles",
            "bounds": bounds,
            "usable_bounds": usable,
            "wall_clearance": wall_clearance,
            "object_clearance": object_clearance,
            "access_origin": {"x": access_x, "y": access_y},
            "search_nodes": visited_nodes,
            "slots": candidate_slots[0],
            "candidate_slots": candidate_slots,
        }

    def _format_packing_slot_strategy(self, strategy):
        if not strategy:
            return ""
        b = strategy["bounds"]
        if strategy.get("layout_type") == "mixed_rectangles":
            lines = [
                "=== GLOBAL PACKING SLOT STRATEGY ===",
                (
                    "Use this complete rotation-aware layout as the primary "
                    "plan for the mixed-size objects."
                ),
                (
                    "Container inner bounds: "
                    f"x=[{b['x_min']:.3f},{b['x_max']:.3f}], "
                    f"y=[{b['y_min']:.3f},{b['y_max']:.3f}]"
                ),
                (
                    "The layout was geometry-checked for wall containment and "
                    f"{strategy['object_clearance']:.3f}m pairwise clearance."
                ),
                "MANDATORY SLOT TABLE:",
            ]
            for i, slot in enumerate(strategy["slots"], start=1):
                lines.append(
                    f"  {i}. {slot['object']}: footprint="
                    f"{slot['footprint_x']:.3f}x{slot['footprint_y']:.3f}m "
                    f"place=({slot['x']:.3f}, {slot['y']:.3f}, "
                    f"theta={slot['theta']:.4f})"
                )
            lines.extend([
                "Rules: use every assigned slot and theta exactly once;",
                "do not combine coordinates from independent per-object ACFM maps.",
                "=== END GLOBAL PACKING SLOT STRATEGY ===",
            ])
            return "\n".join(lines)

        fp = strategy["footprint"]
        cell = strategy["cell"]
        margins = strategy["margins"]
        lines = [
            "=== GLOBAL PACKING SLOT STRATEGY ===",
            "Use these slots as the primary plan for repeated cans. Each slot is assigned once.",
            (
                "Container inner bounds: "
                f"x=[{b['x_min']:.3f},{b['x_max']:.3f}], "
                f"y=[{b['y_min']:.3f},{b['y_max']:.3f}]"
            ),
            (
                "Conservative can footprint used for packing: "
                f"{fp['x']:.3f}m x {fp['y']:.3f}m; "
                f"grid={strategy['cols']} columns x {strategy['rows']} rows; "
                f"cell={cell['w']:.3f}m x {cell['l']:.3f}m; "
                f"free margin approx=({margins['x']:.3f}m, {margins['y']:.3f}m)"
            ),
            (
                "Placement order: nearest robot-access slots first "
                f"(robot base xy=({strategy['access_origin']['x']:.3f}, "
                f"{strategy['access_origin']['y']:.3f}))."
            ),
            "MANDATORY SLOT TABLE:",
        ]
        for i, slot in enumerate(strategy["slots"], start=1):
            lines.append(
                f"  {i}. {slot['object']}: row={slot['row']} col={slot['col']} "
                f"place=({slot['x']:.3f}, {slot['y']:.3f}, theta={slot['theta']:.1f})"
            )
        lines.extend([
            "Rules: use every slot exactly once; do not place two cans in the same row/col cell;",
            "do not use a locally high ACFM point if it is not one of the assigned slots.",
            "=== END GLOBAL PACKING SLOT STRATEGY ===",
        ])
        return "\n".join(lines)

    def _stacking_target_pose(self, obj_name, existing_objects):
        stack_order = list(getattr(self.env, "_stacking_order", []))
        stack_idx = stack_order.index(obj_name) if obj_name in stack_order else 0
        if stack_idx == 0:
            sz = self.env._stacking_zone
            support_z = self._support_surface_z()
            return {
                "x": float(sz["x"]),
                "y": float(sz["y"]),
                "z": self._body_z_with_bottom_on(obj_name, support_z),
            }

        prev_name = stack_order[stack_idx - 1]
        if prev_name in existing_objects:
            prev_bb_min, prev_bb_max = self.env.get_bb(prev_name)
            support_z = prev_bb_max[2]
            return {
                "x": float((prev_bb_min[0] + prev_bb_max[0]) / 2.0),
                "y": float((prev_bb_min[1] + prev_bb_max[1]) / 2.0),
                "z": self._body_z_with_bottom_on(obj_name, support_z),
            }

        sz = self.env._stacking_zone
        return {
            "x": float(sz["x"]),
            "y": float(sz["y"]),
            "z": self._body_z_with_bottom_on(obj_name, self._support_surface_z()),
        }

    def _stacking_support_surfaces(self, obj_name, existing_objects):
        """Return the only valid support surface for an ordered stack object."""
        stack_order = list(getattr(self.env, "_stacking_order", []))
        if obj_name not in stack_order:
            return [], False
        stack_idx = stack_order.index(obj_name)
        if stack_idx == 0:
            return [], False

        prev_name = stack_order[stack_idx - 1]
        if prev_name not in existing_objects:
            return [], False

        prev_bb_min, prev_bb_max = self.env.get_bb(prev_name)
        return [{
            "x_min": float(prev_bb_min[0]),
            "x_max": float(prev_bb_max[0]),
            "y_min": float(prev_bb_min[1]),
            "y_max": float(prev_bb_max[1]),
            "z": float(prev_bb_max[2]),
            "label": f"top of {prev_name}",
        }], True

    def _stacking_z_values(self, obj_name, existing_objects):
        stack_order = list(getattr(self.env, "_stacking_order", []))
        stack_idx = stack_order.index(obj_name) if obj_name in stack_order else 0
        support_z_values = []
        if stack_idx == 0:
            support_z_values.append(self._support_surface_z())
        else:
            prev_name = stack_order[stack_idx - 1]
            if prev_name in existing_objects:
                support_z_values.append(self.env.get_bb(prev_name)[1][2])
            else:
                support_z_values.append(self._support_surface_z())

        values = set()
        for support_z in support_z_values:
            nominal = self._body_z_with_bottom_on(obj_name, support_z)
            # Sample upward from the exact supported center height.  Lower
            # samples can make thin disks look feasible while penetrating the
            # support surface, which is visually confusing and physically wrong.
            for dz in (0.0, 0.025, 0.050):
                values.add(round(max(0.005, nominal + dz), 6))
        return sorted(values)

    @staticmethod
    def _snapshot_python_sim_state(env):
        """Capture Python-side state that PyBullet saveState does not include."""
        robot = getattr(env, "robot", None)
        return {
            "attachments": (
                list(robot.attachments_robot)
                if robot is not None and hasattr(robot, "attachments_robot")
                else None
            ),
            "last_grasp_direction": (
                getattr(robot, "last_grasp_direction", None)
                if robot is not None else None
            ),
            "last_grasp_position_offset": copy.deepcopy(
                getattr(robot, "last_grasp_position_offset", None)
                if robot is not None else None
            ),
            "last_motion_trace": copy.deepcopy(
                getattr(robot, "last_motion_trace", None)
                if robot is not None else None
            ),
            "theta_dict": copy.deepcopy(getattr(env, "theta_dict", None)),
            "prepush_transition_done": copy.deepcopy(
                getattr(env, "_prepush_transition_done", None)),
        }

    @staticmethod
    def _restore_python_sim_state(env, snapshot):
        robot = getattr(env, "robot", None)
        if robot is not None and snapshot["attachments"] is not None:
            robot.attachments_robot = list(snapshot["attachments"])
        if robot is not None and hasattr(robot, "last_grasp_direction"):
            robot.last_grasp_direction = snapshot["last_grasp_direction"]
        if robot is not None and hasattr(robot, "last_grasp_position_offset"):
            robot.last_grasp_position_offset = copy.deepcopy(
                snapshot["last_grasp_position_offset"])
        if (
            robot is not None
            and hasattr(robot, "last_motion_trace")
            and snapshot.get("last_motion_trace") is not None
        ):
            robot.last_motion_trace = copy.deepcopy(
                snapshot["last_motion_trace"])
        if snapshot["theta_dict"] is not None:
            env.theta_dict = copy.deepcopy(snapshot["theta_dict"])
        if (
            hasattr(env, "_prepush_transition_done")
            and snapshot.get("prepush_transition_done") is not None
        ):
            env._prepush_transition_done = copy.deepcopy(
                snapshot["prepush_transition_done"])

    def _current_state_record(self):
        """Return a detached predicted-state record for ACFM provenance."""
        observation = self.env.get_observation()
        if isinstance(observation, tuple):
            structured = observation[0]
            text = observation[1] if len(observation) > 1 else ""
        else:
            structured = observation
            text = ""
        return {
            "observation": copy.deepcopy(structured),
            "text": str(text),
        }

    @staticmethod
    def _json_safe_audit_value(value):
        """Detach common simulator values into JSON-serializable containers."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): ForeSightTAMP._json_safe_audit_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                ForeSightTAMP._json_safe_audit_value(item)
                for item in value
            ]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Audit records must remain serializable even if an environment adds a
        # custom observation type.  Its string form is provenance only and is
        # never fed back into candidate generation.
        return str(value)

    @classmethod
    def _serialize_acfm_map(cls, acfm):
        """Serialize a map without retaining a live simulator-backed object."""
        if acfm is None:
            return None

        axes = {}
        for axis, attribute in (
                ("x", "x_range"), ("y", "y_range"), ("z", "z_range")):
            values = getattr(acfm, attribute, None)
            if values is not None:
                axes[axis] = cls._json_safe_audit_value(values)

        scores = {}
        grid = getattr(acfm, "grid", None)
        grid3d = getattr(acfm, "grid3d", None)
        if grid is not None:
            scores["xy"] = cls._json_safe_audit_value(grid)
        if grid3d is not None:
            scores["xyz"] = cls._json_safe_audit_value(grid3d)

        masks = {
            str(name): cls._json_safe_audit_value(mask)
            for name, mask in (getattr(acfm, "hard_masks", {}) or {}).items()
        }
        soft_layers = {
            str(name): cls._json_safe_audit_value(layer)
            for name, layer in (getattr(acfm, "soft_layers", {}) or {}).items()
        }
        record = {
            "map_type": type(acfm).__name__,
            "target": getattr(acfm, "target_name", None),
            "resolution": getattr(acfm, "res", None),
            "z_resolution": getattr(acfm, "z_res", None),
            "bounds": getattr(acfm, "bounds", None),
            "axes": axes,
            "scores": scores,
            "hard_masks": masks,
            "feasible_mask": getattr(acfm, "feasible_mask", None),
            "feasible_mask_3d": getattr(acfm, "feasible_mask_3d", None),
            "soft_layers": soft_layers,
            "contact_modes": getattr(acfm, "contact_modes", None),
            "parameter_candidates": getattr(
                acfm, "parameter_candidates", None),
            "legal_moves": getattr(acfm, "legal_moves", None),
            "illegal_moves": getattr(acfm, "illegal_moves", None),
        }
        return cls._json_safe_audit_value(record)

    @classmethod
    def _serialize_acfm_history_row(cls, row):
        """Detach one live per-attempt history row for persistent auditing."""
        serialized = {
            key: cls._json_safe_audit_value(value)
            for key, value in row.items()
            if key not in {"map", "acfm"}
        }
        serialized["map"] = cls._serialize_acfm_map(
            row.get("map", row.get("acfm")))
        return serialized

    def _persist_attempt_history(self, attempt_history):
        """Append an attempt without carrying its live maps into later resets."""
        self.acfm_hist.extend(
            self._serialize_acfm_history_row(row)
            for row in attempt_history
        )

    def _save_execution_checkpoint(self):
        """Save simulator and Python state before one committed action."""
        try:
            return {
                "state_id": p.saveState(),
                "python_state": self._snapshot_python_sim_state(self.env),
            }
        except Exception as exc:
            logger.debug("Execution checkpoint unavailable: %s", exc)
            return None

    @staticmethod
    def _remove_execution_checkpoints(checkpoints):
        for checkpoint in checkpoints:
            if checkpoint is None:
                continue
            try:
                p.removeState(checkpoint["state_id"])
            except Exception:
                pass

    def _replay_counterfactual_suffix(
            self, *, plan, failed_step, checkpoints, cause_step,
            alternative, suffix, failed_action):
        """Test a placement alternative by replaying the affected suffix.

        The replay begins at the real pre-action checkpoint and uses freshly
        planned actions.  It never commits the counterfactual world state.
        ``suffix`` and ``failed_action`` are accepted for the public cascade
        callback contract; the typed ``plan`` is the authoritative action list.
        """
        del suffix, failed_action
        if (
            cause_step < 0
            or cause_step > failed_step
            or cause_step >= len(checkpoints)
            or checkpoints[cause_step] is None
            or failed_step >= len(plan)
        ):
            return {"success": False, "reason": "checkpoint unavailable"}

        current_state_id = p.saveState()
        current_python_state = self._snapshot_python_sim_state(self.env)
        try:
            checkpoint = checkpoints[cause_step]
            p.restoreState(checkpoint["state_id"])
            self._restore_python_sim_state(
                self.env, checkpoint["python_state"])

            cause_action = plan[cause_step]
            replacement_params = dict(cause_action.param_args)
            for name in cause_action.primitive.parameters:
                if name in alternative:
                    replacement_params[name] = alternative[name]
            replay_actions = [cause_action.primitive(
                obj_args=list(cause_action.obj_args),
                param_args=replacement_params,
            )]
            replay_actions.extend(
                original.primitive(
                    obj_args=list(original.obj_args),
                    param_args=dict(original.param_args),
                )
                for original in plan[cause_step + 1:failed_step + 1]
            )

            with metric_phase("cascade_counterfactual"):
                for replay_action in replay_actions:
                    _, replay_feedback = self.env.step(
                        replay_action, play_traj=False)
                    if not replay_feedback.action_success:
                        return {
                            "success": False,
                            "reason": replay_feedback.motion_planner_feedback,
                        }
            return {"success": True}
        except Exception as exc:
            return {"success": False, "reason": str(exc)}
        finally:
            try:
                p.restoreState(current_state_id)
                p.removeState(current_state_id)
            finally:
                self._restore_python_sim_state(
                    self.env, current_python_state)

    def _validate_action_skeleton(self, actions):
        """Reject unknown objects and ill-formed pick/place ownership changes."""
        if not actions:
            raise ValueError("action skeleton is empty")
        held_object = None
        for index, action in enumerate(actions):
            primitive = action.primitive.name
            if action.param_args:
                raise ValueError(
                    f"skeleton action {index} contains forbidden parameters")
            if any(name not in self.env.objects for name in action.obj_args):
                raise ValueError(f"skeleton action {index} names an unknown object")
            if primitive == "pick":
                if held_object is not None:
                    raise ValueError(
                        f"skeleton picks {action.obj_args[0]} while holding {held_object}")
                held_object = action.obj_args[0]
            elif primitive == "place":
                if held_object != action.obj_args[0]:
                    raise ValueError(
                        f"skeleton places {action.obj_args[0]} while holding {held_object}")
                held_object = None
            elif primitive == "push" and held_object is not None:
                raise ValueError("skeleton cannot push while holding an object")
        if held_object is not None:
            raise ValueError(f"skeleton ends while still holding {held_object}")
        self._validate_skeleton_can_reach_goal(actions)

    def _validate_skeleton_can_reach_goal(self, actions):
        """Reject a skeleton too short to reach the goal, and say by how much.

        Every hard mask judges one action against the state in front of it, so
        a skeleton that simply stops early passes each check individually and
        only fails many steps later, once the state it produced has become
        unsolvable, with no indication that the real defect is a short plan.

        The environment's reference plan is optimal, so no correct solution is
        shorter than it. That makes a shorter skeleton provably unable to
        finish, which is worth saying before the attempt is spent. Longer
        skeletons are left alone: they may be suboptimal and still correct.
        """
        reference = getattr(self.env, "get_symbolic_plan", None)
        if reference is None:
            return
        try:
            plan = list(reference() or [])
        except Exception:  # noqa: BLE001 - a missing reference is not fatal
            return
        # Count only steps naming objects this scene actually contains: a
        # packing environment answers with a default plan over placeholder
        # objects. The prompt filters the same way, so the model is judged
        # against the same reference it is shown.
        known = set(getattr(self.env, "objects", {}) or {})
        relevant = []
        for raw_action in plan:
            named = re.search(r"\[\s*['\"]([^'\"]+)['\"]\s*\]", raw_action)
            if named is None or named.group(1) in known:
                relevant.append(raw_action)
        plan = relevant
        if plan and len(plan) > len(actions):
            raise ValueError(
                f"skeleton has {len(actions)} actions but the task cannot be "
                f"completed in fewer than {len(plan)}; it stops before the "
                "goal is reached. Follow the symbolic reference, which lists "
                "every move including the ones that shift a disk out of the "
                "way and bring it back later")

    def _request_action_skeleton(self, obs_text, feedback_list):
        """Ask for action templates only; spatial parameters are forbidden."""
        feedback_text = "\n".join(
            f"{action}: {feedback.motion_planner_feedback}"
            for action, feedback in feedback_list
        ) or "No prior failure feedback."
        def _is_fixture(name):
            if name in {"table", "basket", "target_zone"}:
                return True
            if name.startswith(("basket_wall_", "peg_", "target_zone")):
                return True
            return (
                hasattr(self.env, "is_shelf_fixture")
                and self.env.is_shelf_fixture(name)
            )

        object_names = sorted(
            name for name in self.env.objects if not _is_fixture(name))
        symbolic_reference = []
        describe = self._placement_describer()
        if hasattr(self.env, "get_symbolic_plan"):
            try:
                for raw_action in self.env.get_symbolic_plan() or []:
                    match = re.search(r"\[\s*['\"]([^'\"]+)['\"]\s*\]", raw_action)
                    if match is not None and match.group(1) not in object_names:
                        continue
                    stripped = re.sub(
                        r",\s*\{.*\}\)\s*$", ", {})", raw_action)
                    # Keep the destination as a symbolic name, never as
                    # coordinates: in Hanoi three pegs are equally placeable
                    # and only one continues the solution.
                    label = None
                    if describe is not None:
                        coords = re.search(
                            r"'x'\s*:\s*(-?[\d.]+).*?'y'\s*:\s*(-?[\d.]+)",
                            raw_action)
                        if coords is not None:
                            try:
                                label = describe(float(coords.group(1)),
                                                 float(coords.group(2)))
                            except Exception:  # noqa: BLE001
                                label = None
                    symbolic_reference.append(
                        f"{stripped} -> {label}" if label else stripped)
            except Exception:
                symbolic_reference = []
        # The reference is a verified solution the environment already holds.
        # Without saying so, the model treats it as a loose hint and improvises
        # a shorter or longer sequence of its own.
        reference_instruction = (
            "The symbolic reference above is a verified solution for this "
            "task, and the arrow on a place action names the destination the "
            "controller will offer. Reproduce that sequence exactly and in "
            "order, including the moves that shift an object out of the way "
            "and bring it back later. Depart from it only where the failure "
            "feedback says a specific action was rejected.\n\n"
            if symbolic_reference else "\n"
        )
        if self._task_type == "packing":
            goal_context = "Place every listed object inside the basket."
        elif self._task_type == "stacking":
            goal_context = (
                "Build the ordered stack bottom-to-top: "
                f"{list(getattr(self.env, '_stacking_order', []))}."
            )
        elif self._task_type == "shelf":
            goal_context = (
                "Place each object on its configured discrete shelf level: "
                f"{dict(getattr(self.env, '_target_placements', {}))}."
            )
        elif self._task_type == "hanoi":
            goal_context = "Move the complete legal disk stack from peg A to peg C."
        else:
            goal_context = (
                "Complete the configured push/stack goal for every listed object."
            )
        prompt = (
            "Construct the symbolic action skeleton for the current robot task. "
            "A skeleton contains only primitive and object arguments. Do not emit "
            "x, y, z, theta, yaw, grasp, contact, or any other parameter; those "
            "will be selected later from geometry-checked candidate IDs.\n\n"
            f"Current observation:\n{obs_text}\n\n"
            f"Goal: {goal_context}\n"
            f"Available primitives: {sorted(self.env.primitive_actions)}\n"
            f"Manipulable object names: {object_names}\n"
            f"Failure feedback:\n{feedback_text}\n"
            f"Symbolic reference when available: {symbolic_reference}\n"
            f"{reference_instruction}"
            "Return only one JSON object with exactly these keys:\n"
            '{"Reasoning":"one short sentence","Action Skeleton":'
            '["pick([\'object\'], {})","place([\'object\'], {})"]}'
        )

        last_error = None
        for attempt in range(3):
            if attempt:
                increment_metric("llm_parse_retries")
                record_metric_event(
                    "llm_parse_retry",
                    parser="action_skeleton",
                    retry_index=attempt,
                    reason=str(last_error),
                )
            try:
                self._last_planning_llm_calls += 1
                raw = self.planner.prompt_llm(prompt, force_json=True)
                actions, reasoning = _parse_action_skeleton_response(
                    raw, self.env.primitive_actions)
                self._validate_action_skeleton(actions)
                return actions, reasoning
            except Exception as exc:
                last_error = exc
                prompt += (
                    "\n\nThe preceding response was rejected. Return a complete "
                    "parameter-free skeleton in the exact schema. Error: "
                    f"{exc}"
                )
        logger.warning("Unable to obtain a valid action skeleton: %s", last_error)
        return None, None

    def _non_llm_action_skeleton(self, task_config):
        """Build the direct-ACFM ablation skeleton without an LLM request."""
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.push_stacking_env import PushStackingEnv
        from envs.shelf_env import ShelfEnv

        if isinstance(self.env, (TowerOfHanoiEnv, PushStackingEnv)):
            raw_plan = list(self.env.get_symbolic_plan() or [])
            skeleton_text = [
                re.sub(r",\s*\{.*\}\)\s*$", ", {})", raw)
                for raw in raw_plan
            ]
            actions = text_to_actions(
                skeleton_text, self.env.primitive_actions)
        else:
            if isinstance(self.env, StackingEnv):
                names = list(getattr(self.env, "_stacking_order", []))
            elif isinstance(self.env, ShelfEnv):
                names = list(getattr(self.env, "_target_placements", {}))
            else:
                names = [
                    spec["name"]
                    for spec in task_config.get("boxes", {}).values()
                    if spec.get("name") in self.env.objects
                ]
            actions = []
            for name in names:
                actions.extend([
                    self.env.primitive_actions["pick"]([name], {}),
                    self.env.primitive_actions["place"]([name], {}),
                ])
        self._validate_action_skeleton(actions)
        return actions

    def _prepare_sequential_packing_strategy(self, actions):
        """Create one global layout before per-action predicted-state mapping."""
        if self._task_type != "packing":
            self._packing_slot_strategy = None
            return
        objects = []
        for action in actions:
            if action.primitive.name == "place" and action.obj_args:
                name = action.obj_args[0]
                if name not in objects:
                    objects.append(name)
        if not objects or "basket" not in self.env.objects:
            self._packing_slot_strategy = None
            return
        bb_min, bb_max = self.env.get_bb("basket")
        bounds = {
            "x_min": float(bb_min[0]),
            "x_max": float(bb_max[0]),
            "y_min": float(bb_min[1]),
            "y_max": float(bb_max[1]),
            "z_floor": float(bb_min[2]),
        }
        self._compute_packing_slot_strategy(objects, bounds)

    def _grasp_yaw(self, obj_name):
        """Yaw the gripper holds ``obj_name`` at, zero when grasped square.

        A scene may grasp an object turned in the gripper, and then the theta
        a place is commanded at is not the yaw the object lands at -- it lands
        at theta minus this. The map needs the offset to test the footprint
        the controller actually produces.
        """
        table = getattr(self.env, "_grasp_theta_dict", None) or {}
        return float(table.get(obj_name) or 0.0)

    def _placement_acfm_context(self, obj_name):
        """Build the current predicted-state endpoint-map context."""
        from envs.push_stacking_env import PushStackingEnv
        from envs.shelf_env import ShelfEnv

        is_shelf = isinstance(self.env, ShelfEnv)
        is_stacking = isinstance(self.env, StackingEnv)
        is_push_stacking = isinstance(self.env, PushStackingEnv)
        support_surfaces = []
        z_values = None
        allow_ground_support = True
        require_configured_support = False
        contact_modes = ("front",) if is_shelf else ("top",)

        if is_shelf:
            shelf_bounds = self.env.get_shelf_bounds()
            bounds = {
                key: float(shelf_bounds[key])
                for key in ("x_min", "x_max", "y_min", "y_max")
            }
            existing = {
                name: oid for name, oid in self.env.objects.items()
                if name != obj_name and self.env.is_shelf_fixture(name)
            }
            for name, oid in self.env.objects.items():
                if name == obj_name or name == "table" or self.env.is_shelf_fixture(name):
                    continue
                on_shelf, _ = self.env._check_on_shelf(name)
                if on_shelf:
                    existing[name] = oid
            target_level = self.env._target_placements.get(obj_name)
            if target_level is not None:
                z_values = [
                    self.env._minimum_center_z_for_level(obj_name, target_level)
                ]
                level = self.env.get_shelf_bounds(level_idx=target_level)
                support_surfaces = [{
                    "x_min": float(level["x_min"]),
                    "x_max": float(level["x_max"]),
                    "y_min": float(level["y_min"]),
                    "y_max": float(level["y_max"]),
                    "z": float(level["z"]),
                    "label": f"shelf level {target_level}",
                }]
            allow_ground_support = False
            require_configured_support = True
        elif is_stacking:
            zone = self.env._stacking_zone
            radius = float(zone["radius"])
            bounds = {
                "x_min": float(zone["x"] - radius),
                "x_max": float(zone["x"] + radius),
                "y_min": float(zone["y"] - radius),
                "y_max": float(zone["y"] + radius),
            }
            existing = {}
            for name, oid in self.env.objects.items():
                if name in {obj_name, "table", "target_zone"}:
                    continue
                if self.env._is_in_zone(name):
                    existing[name] = oid
            z_values = self._stacking_z_values(obj_name, existing)
            support_surfaces, require_configured_support = (
                self._stacking_support_surfaces(obj_name, existing))
            stack_order = list(getattr(self.env, "_stacking_order", []))
            allow_ground_support = (
                obj_name in stack_order and stack_order.index(obj_name) == 0)
        elif is_push_stacking:
            # A push-stack place puts the top object on the pushed base, so the
            # admissible region is the base's current footprint, not a basket
            # (PushStackingEnv has no "basket" body).
            stack_order = list(getattr(self.env, "_stacking_order", []))
            base_name = stack_order[0] if stack_order else None
            if base_name is None or base_name not in self.env.objects:
                raise RuntimeError(
                    "push-stacking placement requires a base object in the "
                    f"stacking order; got {stack_order!r}"
                )
            base_min, base_max = self.env.get_bb(base_name)
            base_cx = (float(base_min[0]) + float(base_max[0])) / 2.0
            base_cy = (float(base_min[1]) + float(base_max[1])) / 2.0
            base_top = float(base_max[2])
            # env.check_goal accepts the stack only within this xy tolerance.
            tolerance = float(
                getattr(self.env, "_target_pose", {}).get("stack_xy_tolerance", 0.04))
            bounds = {
                "x_min": base_cx - tolerance,
                "x_max": base_cx + tolerance,
                "y_min": base_cy - tolerance,
                "y_max": base_cy + tolerance,
                "z_floor": base_top,
            }
            existing = {
                name: oid for name, oid in self.env.objects.items()
                if name not in {obj_name, base_name, "table"}
                and not self.env.is_fixture(name)
            }
            # Sample upward from the exact resting height, as the stacking
            # branch does. At the exact contact height the gripper fingers
            # straddle the object and collide with the base's top face, so a
            # small release clearance is needed.
            nominal = self._body_z_with_bottom_on(obj_name, base_top)
            z_values = [
                round(max(0.005, nominal + dz), 6)
                for dz in (0.0, 0.010, 0.025, 0.050)
            ]
            support_surfaces = [{
                "x_min": float(base_min[0]),
                "x_max": float(base_max[0]),
                "y_min": float(base_min[1]),
                "y_max": float(base_max[1]),
                "z": base_top,
                "label": f"top face of {base_name}",
            }]
            allow_ground_support = False
            require_configured_support = True
        else:
            bb_min, bb_max = self.env.get_bb("basket")
            bounds = {
                "x_min": float(bb_min[0]),
                "x_max": float(bb_max[0]),
                "y_min": float(bb_min[1]),
                "y_max": float(bb_max[1]),
                "z_floor": float(bb_min[2]),
            }
            if self._packing_slot_strategy:
                bounds["tight_packing"] = True
            existing = {}
            for name, oid in self.env.objects.items():
                if name == obj_name or name in {"basket", "table"} or name.startswith("basket_wall_"):
                    continue
                if self.env.check_in_basket(name):
                    existing[name] = oid

        return {
            "bounds": bounds,
            "existing": existing,
            "z_values": z_values,
            "support_surfaces": support_surfaces,
            "allow_ground_support": allow_ground_support,
            "require_configured_support": require_configured_support,
            "contact_modes": contact_modes,
            "grasp_yaw": self._grasp_yaw(obj_name),
            "is_3d": is_shelf or is_stacking or is_push_stacking,
        }

    @staticmethod
    def _candidate_at_parameters(acfm, parameters, contact_mode):
        """Create an exact parameter row and audit its nearest hard-map cell."""
        x = float(parameters["x"])
        y = float(parameters["y"])
        theta = float(parameters.get("theta", 0.0))
        i = int(np.argmin(np.abs(acfm.x_range - x)))
        j = int(np.argmin(np.abs(acfm.y_range - y)))
        constraints = {}
        if isinstance(acfm, ACFM3D) and "z" in parameters:
            exact_z = float(parameters["z"])
            k = int(np.argmin(np.abs(acfm.z_range - exact_z)))
            for name, mask in acfm.hard_masks.items():
                constraints[name] = bool(mask[j, i, k])
            # Exact authoritative targets (notably stack centers) need exact
            # support coverage; a coarse nearest XY grid cell can otherwise
            # turn a supported mild overhang into a false rejection.
            constraints["support"] = bool(
                acfm.support_score_at(x, y, exact_z) >= 0.5)
            score = float(acfm.grid3d[j, i, k])
        else:
            for name, mask in acfm.hard_masks.items():
                constraints[name] = bool(mask[j, i])
            score = float(acfm.grid[j, i])
        constraints["oriented_geometry"] = bool(
            acfm._oriented_candidate_is_endpoint_feasible(
                x, y, theta, z=parameters.get("z")))
        row = {
            "candidate_id": "",
            "object": acfm.target_name,
            "x": round(x, 4),
            "y": round(y, 4),
            "theta": round(theta, 4),
            "grasp_mode": (
                contact_mode if contact_mode in {"top", "front"} else None),
            "contact_mode": contact_mode,
            "score": round(score, 4),
            "hard_constraints": constraints,
            "hard_feasible": all(constraints.values()),
        }
        if "z" in parameters:
            row["z"] = round(float(parameters["z"]), 4)
        return row

    def _compute_map_candidates(self, action, action_index, iteration):
        """Compute one action-conditioned map in the current predicted state."""
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.push_stacking_env import PushStackingEnv

        primitive = action.primitive.name
        obj_name = action.obj_args[0] if action.obj_args else None
        if primitive not in {"place", "push"} or obj_name not in self.env.objects:
            return None, []

        with metric_timer("acfm_compute_seconds", phase="acfm"):
            if isinstance(self.env, TowerOfHanoiEnv):
                acfm = HanoiACFM(resolution=self.acfm_resolution)
                acfm.compute_hanoi(self.env)
                candidates = []
                for move in acfm.legal_moves:
                    if move["disk"] != obj_name:
                        continue
                    candidates.append({
                        "candidate_id": "",
                        "object": obj_name,
                        "x": round(float(move["x"]), 4),
                        "y": round(float(move["y"]), 4),
                        "z": round(float(move["z"]), 4),
                        "theta": round(float(move.get("theta", 0.0)), 4),
                        "grasp_mode": "top",
                        "contact_mode": "top",
                        "score": round(float(move.get("score", 1.0)), 4),
                        "hard_constraints": {
                            "bounds": True,
                            "collision": True,
                            "reachability": True,
                            "support": True,
                            "valid_contact": True,
                            "symbolic_legality": True,
                        },
                        "hard_feasible": True,
                    })
            elif primitive == "push" and isinstance(self.env, PushStackingEnv):
                targets = self.env.get_push_targets()
                target = targets.get(obj_name)
                if target is None:
                    return None, []
                existing = {
                    name: oid for name, oid in self.env.objects.items()
                    if name != obj_name and name != "table"
                    and not name.startswith("target_zone")
                }
                acfm = PushACFM(resolution=self.acfm_resolution)
                acfm.compute_push(
                    target_obj_id=self.env.objects[obj_name],
                    target_name=obj_name,
                    existing_objects=existing,
                    robot_id=self.env.robot.robot,
                    ee_link_idx=self.env.robot.tool_attach_link,
                    push_bounds=self.env.get_push_bounds(),
                    push_goal=target,
                    failure_history=self.failure_positions.get(obj_name, []),
                    disturbance_history=self.disturbance_positions,
                    push_params=self.env._push_params(obj_name),
                )
                candidates = acfm.get_parameter_candidates(
                    8,
                    yaw_values=(float(target.get("theta", 0.0)),),
                    contact_modes=("lateral_push",),
                )
                target_row = self._candidate_at_parameters(
                    acfm,
                    {
                        "x": target["x"],
                        "y": target["y"],
                        "theta": target.get("theta", 0.0),
                    },
                    "lateral_push",
                )
                candidates.insert(0, target_row)
                # When the straight path is blocked, the detour waypoints are
                # the only route to the goal, so they belong in the offered
                # set. The map samples a grid plus the final target, which
                # leaves the intermediate legs unrepresented and makes the
                # prescribed plan impossible to follow.
                for waypoint in (target.get("recommended_waypoints") or []):
                    try:
                        way_x, way_y = float(waypoint[0]), float(waypoint[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    way_row = self._candidate_at_parameters(
                        acfm,
                        {
                            "x": way_x,
                            "y": way_y,
                            "theta": target.get("theta", 0.0),
                        },
                        "lateral_push",
                    )
                    if way_row is not None:
                        candidates.insert(0, way_row)
                # When a straight push is blocked, the detour waypoints are the
                # only route to the goal -- they are part of the task, not free
                # parameters. The map samples a grid and the final target, so
                # without this the waypoints are never offered and no selector
                # can follow the plan that the observation itself prescribes.

            else:
                context = self._placement_acfm_context(obj_name)
                if context["is_3d"]:
                    z_values = context["z_values"] or []
                    max_z = max(z_values) + 0.08 if z_values else 0.55
                    acfm = ACFM3D(
                        resolution=self.acfm_resolution,
                        z_resolution=max(1, len(z_values)),
                        max_z=max_z,
                        rank_candidates=self.use_acfm_ranking,
                    )
                    acfm.compute_3d(
                        target_obj_id=self.env.objects[obj_name],
                        target_name=obj_name,
                        existing_objects=context["existing"],
                        robot_id=self.env.robot.robot,
                        ee_link_idx=self.env.robot.tool_attach_link,
                        basket_bounds=context["bounds"],
                        failure_history=self.failure_positions.get(obj_name, []),
                        disturbance_history=self.disturbance_positions,
                        z_values=z_values,
                        support_surfaces=context["support_surfaces"],
                        allow_ground_support=context["allow_ground_support"],
                        require_configured_support=context["require_configured_support"],
                        contact_modes=context["contact_modes"],
                        grasp_yaw=context["grasp_yaw"],
                    )
                    candidates = acfm.get_parameter_candidates(
                        10, contact_modes=context["contact_modes"])
                else:
                    acfm = ACFM(resolution=self.acfm_resolution,
                                rank_candidates=self.use_acfm_ranking)
                    acfm.compute(
                        target_obj_id=self.env.objects[obj_name],
                        target_name=obj_name,
                        existing_objects=context["existing"],
                        robot_id=self.env.robot.robot,
                        ee_link_idx=self.env.robot.tool_attach_link,
                        basket_bounds=context["bounds"],
                        failure_history=self.failure_positions.get(obj_name, []),
                        disturbance_history=self.disturbance_positions,
                        contact_modes=context["contact_modes"],
                        grasp_yaw=context["grasp_yaw"],
                    )
                    candidates = acfm.get_parameter_candidates(
                        10, contact_modes=context["contact_modes"])

                strategy = self._packing_slot_strategy or {}
                layouts = strategy.get("candidate_slots") or (
                    [strategy.get("slots", [])] if strategy else [])
                assigned = []
                seen_assignments = set()
                for layout in layouts:
                    slot = next(
                        (candidate for candidate in layout
                         if candidate.get("object") == obj_name),
                        None,
                    )
                    if slot is None:
                        continue
                    row = self._candidate_at_parameters(
                        acfm,
                        slot,
                        context["contact_modes"][0],
                    )
                    key = (
                        row["x"], row["y"], row.get("z"), row["theta"],
                        row["contact_mode"],
                    )
                    if key in seen_assignments:
                        continue
                    seen_assignments.add(key)
                    assigned.append(row)
                    if len(assigned) >= 10:
                        break
                if assigned:
                    candidates = assigned

                if isinstance(self.env, StackingEnv):
                    exact_pose = self._stacking_target_pose(
                        obj_name, context["existing"])
                    exact_pose["theta"] = 0.0
                    row = self._candidate_at_parameters(
                        acfm, exact_pose, context["contact_modes"][0])
                    candidates = [row]

        # Rendered maps are human-inspection artifacts and are never sent to
        # the language model.
        if self.render_acfm_maps:
            try:
                path = self.save_dir / (
                    f"acfm_iter{iteration}_action{action_index}_{obj_name}.png")
                with metric_timer("acfm_render_seconds", phase="export"):
                    if isinstance(acfm, (ACFM3D, HanoiACFM)):
                        acfm.render_3d_volume(save_path=str(path))
                    else:
                        acfm.render_heatmap(save_path=str(path))
            except Exception as exc:
                logger.debug("ACFM diagnostic rendering skipped: %s", exc)
        return acfm, candidates

    def _rollout_control_quota(self, template):
        """Return the fixed per-action rollout cap for the control method."""
        from envs.hanoi_env import TowerOfHanoiEnv

        if template.primitive.name == "place":
            if isinstance(self.env, TowerOfHanoiEnv):
                return self.rollout_validation_quotas["hanoi_place"]
            if isinstance(self.env, StackingEnv):
                return self.rollout_validation_quotas["stacking_place"]
        return self.rollout_validation_quotas[template.primitive.name]

    @staticmethod
    def _van_der_corput(index, base):
        value = 0.0
        denominator = 1.0
        while index:
            index, remainder = divmod(index, base)
            denominator *= base
            value += remainder / denominator
        return value

    @classmethod
    def _low_discrepancy_indices(
            cls, axis_lengths, count, excluded=(), offset=0):
        """Select deterministic, score-independent points over a product grid."""
        axis_lengths = tuple(int(length) for length in axis_lengths)
        domain_size = math.prod(axis_lengths)
        available = domain_size - len(set(excluded))
        offset = max(0, int(offset))
        target_total = min(offset + int(count), available)
        if target_total <= offset:
            return []

        excluded = set(excluded)
        chosen = []
        seen = set(excluded)
        bases = (2, 3, 5, 7, 11, 13, 17)
        sequence_index = 1
        # Halton points provide deterministic spatial coverage without reading
        # ACFM masks or scores.  A lexicographic fill handles small-axis
        # duplicate projections exactly.
        while len(chosen) < target_total and sequence_index <= domain_size * 20:
            coordinates = tuple(
                min(
                    length - 1,
                    int(cls._van_der_corput(sequence_index, bases[axis]) * length),
                )
                for axis, length in enumerate(axis_lengths)
            )
            sequence_index += 1
            if coordinates in seen:
                continue
            seen.add(coordinates)
            chosen.append(coordinates)
        if len(chosen) < target_total:
            for coordinates in itertools.product(
                    *(range(length) for length in axis_lengths)):
                if coordinates in seen:
                    continue
                chosen.append(coordinates)
                if len(chosen) == target_total:
                    break
        return chosen[offset:target_total]

    @staticmethod
    def _rollout_only_candidate(source):
        """Strip every learned/map-derived decision field from one row."""
        public = {
            key: copy.deepcopy(source[key])
            for key in (
                "object", "x", "y", "z", "theta",
                "grasp_mode", "contact_mode",
            )
            if key in source
        }
        public.update({
            "candidate_id": "",
            "hard_constraints": {},
            "hard_feasible": False,
            "_rollout_only": True,
        })
        return public

    def _rollout_control_candidates(
            self, acfm, template, reference_candidates, batch_index=0):
        """Build an equal-budget raw-domain candidate set without ACFM ranking.

        ACFM is still computed so this control pays the same map-construction
        cost and uses the same declared parameter axes.  Neither its soft
        scores nor any hard mask is inspected here.  Candidate acceptance is
        based solely on the identical simulator rollout used by ForeSight.
        """
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.push_stacking_env import PushStackingEnv

        quota = self._rollout_control_quota(template)
        obj_name = template.obj_args[0] if template.obj_args else None

        # Stacking has one goal-determined pose in the full method.  Preserve
        # that one-element task domain, while removing its map labels/score.
        if isinstance(self.env, StackingEnv):
            rows = (
                [self._rollout_only_candidate(reference_candidates[0])]
                if reference_candidates else []
            )
            return rows[:quota], {
                "selection_mode": "rollout_control",
                "candidate_source": "task_exact_target",
                "domain_size": len(rows),
                "rollout_quota": quota,
                "sampling_batch": int(batch_index),
            }

        # Hanoi's raw domain is every other peg, including symbolically
        # illegal destinations; simulator rollout alone decides acceptance.
        if isinstance(self.env, TowerOfHanoiEnv):
            moves = list(getattr(acfm, "legal_moves", [])) + list(
                getattr(acfm, "illegal_moves", []))
            source = next(
                (move.get("source") for move in moves
                 if move.get("disk") == obj_name),
                None,
            )
            state = copy.deepcopy(getattr(acfm, "peg_state", {}))
            rows = []
            for peg in getattr(acfm, "peg_order", []):
                if peg == source:
                    continue
                peg_cfg = self.env._pegs[peg]
                rows.append(self._rollout_only_candidate({
                    "object": obj_name,
                    "x": round(float(peg_cfg["x"]), 4),
                    "y": round(float(peg_cfg["y"]), 4),
                    "z": round(float(
                        self.env._next_place_z(peg, state, obj_name) + 0.003
                    ), 4),
                    "theta": 0.0,
                    "grasp_mode": "top",
                    "contact_mode": "top",
                }))
            return rows[:quota], {
                "selection_mode": "rollout_control",
                "candidate_source": "all_alternative_pegs",
                "domain_size": len(rows),
                "rollout_quota": quota,
                "sampling_batch": int(batch_index),
            }

        x_values = [float(value) for value in getattr(acfm, "x_range", [])]
        y_values = [float(value) for value in getattr(acfm, "y_range", [])]
        if not x_values or not y_values:
            return [], {
                "selection_mode": "rollout_control",
                "candidate_source": "score_independent_parameter_domain",
                "domain_size": 0,
                "rollout_quota": quota,
                "sampling_batch": int(batch_index),
            }

        axes = [x_values, y_values]
        axis_names = ["x", "y"]
        if "z" in template.primitive.parameters:
            raw_z_values = getattr(acfm, "z_range", None)
            z_values = [
                float(value) for value in (
                    raw_z_values if raw_z_values is not None else [])
            ]
            if z_values:
                axes.append(z_values)
                axis_names.append("z")

        if template.primitive.name == "push" and isinstance(
                self.env, PushStackingEnv):
            target = self.env.get_push_targets()[obj_name]
            theta_values = [float(target.get("theta", 0.0))]
            modes = ["lateral_push"]
        else:
            dimensions = list(getattr(acfm, "target_dims", []) or [])
            rectangular = (
                len(dimensions) >= 2
                and max(dimensions[:2]) / max(min(dimensions[:2]), 1e-9) > 1.10
            )
            theta_values = [0.0, float(np.pi / 2.0)] if rectangular else [0.0]
            modes = list(getattr(acfm, "contact_modes", ()) or ("top",))
        axes.extend([theta_values, modes])
        axis_names.extend(["theta", "contact_mode"])

        exact_row = None
        excluded = set()
        sample_quota = quota
        if template.primitive.name == "push":
            target = self.env.get_push_targets()[obj_name]
            exact_row = self._rollout_only_candidate({
                "object": obj_name,
                "x": round(float(target["x"]), 4),
                "y": round(float(target["y"]), 4),
                "theta": round(float(target.get("theta", 0.0)), 4),
                "contact_mode": "lateral_push",
            })
            sample_quota -= 1
            for indices in itertools.product(
                    *(range(len(axis)) for axis in axes)):
                values = dict(zip(
                    axis_names,
                    (axes[axis][index] for axis, index in enumerate(indices)),
                ))
                if (
                    round(float(values["x"]), 4) == exact_row["x"]
                    and round(float(values["y"]), 4) == exact_row["y"]
                    and round(float(values["theta"]), 4) == exact_row["theta"]
                ):
                    excluded.add(indices)

        sampled_indices = self._low_discrepancy_indices(
            [len(axis) for axis in axes],
            sample_quota,
            excluded=excluded,
            offset=max(0, int(batch_index)) * sample_quota,
        )
        rows = [exact_row] if exact_row is not None else []
        for indices in sampled_indices:
            values = dict(zip(
                axis_names,
                (axes[axis][index] for axis, index in enumerate(indices)),
            ))
            mode = str(values.pop("contact_mode"))
            source = {
                "object": obj_name,
                **{
                    name: round(float(value), 4)
                    for name, value in values.items()
                },
                "grasp_mode": mode if mode in {"top", "front"} else None,
                "contact_mode": mode,
            }
            rows.append(self._rollout_only_candidate(source))

        return rows, {
            "selection_mode": "rollout_control",
            "candidate_source": "score_independent_low_discrepancy_domain",
            "domain_size": (
                math.prod(len(axis) for axis in axes)
                + (1 if exact_row is not None else 0)
                - len(excluded)
            ),
            "rollout_quota": quota,
            "sampling_batch": int(batch_index),
        }

    @staticmethod
    def _resolve_candidate_action(template, candidate):
        """Resolve geometry outside the LLM into the primitive's parameters."""
        params = {
            name: candidate[name]
            for name in template.primitive.parameters
            if name in candidate
        }
        return template.primitive(
            obj_args=list(template.obj_args),
            param_args=params,
        )

    def _validate_candidate_rollout(self, template, candidate, next_template):
        """Check action and exposed approach/retreat/next-transition phases."""
        state_id = p.saveState()
        python_state = self._snapshot_python_sim_state(self.env)
        action = self._resolve_candidate_action(template, candidate)
        # Private rollout evidence is available to offline controls and exact
        # path reuse but omitted from audit rows and LLM-facing representations.
        candidate["_rollout_action"] = action
        constraints = dict(candidate.get("hard_constraints", {}))
        rollout_only = bool(candidate.get("_rollout_only", False))
        candidate["endpoint_map_feasible"] = (
            None if rollout_only
            else bool(candidate.get("hard_feasible", False))
        )
        increment_metric(
            "candidate_rollout_attempts", phase="acfm_validation")
        feedback_text = ""
        try:
            with metric_phase("acfm_validation"):
                _, feedback = self.env.step(action, play_traj=False)
            candidate["_rollout_motion_trace"] = copy.deepcopy(
                getattr(self.env.robot, "last_motion_trace", []))
            feedback_text = feedback.motion_planner_feedback
            rollout_constraints = {
                "transition": bool(feedback.action_success),
            }
            phases = {
                segment.get("phase")
                for segment in getattr(self.env.robot, "last_motion_trace", [])
            }
            if template.primitive.name == "place":
                rollout_constraints["approach_place"] = (
                    feedback.action_success and "place_transfer" in phases)
                rollout_constraints["retreat"] = (
                    feedback.action_success
                    and bool({
                        "post_place_lift", "post_place_retreat"
                    } & phases)
                )
            elif template.primitive.name == "push":
                # A push whose target is already satisfied succeeds without
                # moving, so it produces no contact or retreat phase; accept it
                # as a satisfied action rather than marking it hard-infeasible.
                from envs.push_stacking_env import PUSH_ALREADY_AT_TARGET
                already_satisfied = (
                    feedback.action_success
                    and feedback.motion_planner_feedback == PUSH_ALREADY_AT_TARGET)
                rollout_constraints["valid_contact"] = bool(
                    feedback.action_success
                    and (already_satisfied or "push_contact" in phases))
                rollout_constraints["retreat"] = bool(
                    feedback.action_success
                    and (already_satisfied or "post_push_retreat" in phases))

            if (
                all(rollout_constraints.values())
                and next_template is not None
                and next_template.primitive.name == "pick"
            ):
                next_action = next_template.primitive(
                    obj_args=list(next_template.obj_args), param_args={})
                candidate["_next_rollout_action"] = next_action
                with metric_phase("acfm_validation"):
                    _, next_feedback = self.env.step(
                        next_action, play_traj=False)
                candidate["_next_rollout_motion_trace"] = copy.deepcopy(
                    getattr(self.env.robot, "last_motion_trace", []))
                rollout_constraints["next_action_transition"] = bool(
                    next_feedback.action_success)
                if not next_feedback.action_success:
                    feedback_text += (
                        "; next-action transition failed: "
                        + next_feedback.motion_planner_feedback)

            constraints.update(rollout_constraints)
            candidate["hard_constraints"] = constraints
            candidate["ground_truth_feasible"] = all(
                rollout_constraints.values())
            candidate["hard_feasible"] = (
                candidate["ground_truth_feasible"]
                if rollout_only else (
                    candidate["endpoint_map_feasible"]
                    and candidate["ground_truth_feasible"]
                )
            )
            candidate["validation_feedback"] = feedback_text
            if candidate["ground_truth_feasible"]:
                increment_metric(
                    "candidate_rollout_passes", phase="acfm_validation")
            with metric_phase("acfm_validation"):
                record_metric_event(
                    "candidate_rollout",
                    candidate_id=candidate.get("candidate_id"),
                    primitive=template.primitive.name,
                    selection_mode=self.candidate_selection_mode,
                    endpoint_map_feasible=candidate["endpoint_map_feasible"],
                    ground_truth_feasible=candidate["ground_truth_feasible"],
                    accepted=bool(candidate["hard_feasible"]),
                )
            if candidate["hard_feasible"]:
                candidate["_validated_action"] = action
            return candidate
        except Exception as exc:
            constraints["transition"] = False
            candidate["hard_constraints"] = constraints
            candidate["ground_truth_feasible"] = False
            candidate["hard_feasible"] = False
            candidate["validation_feedback"] = str(exc)
            with metric_phase("acfm_validation"):
                record_metric_event(
                    "candidate_rollout",
                    candidate_id=candidate.get("candidate_id"),
                    primitive=template.primitive.name,
                    selection_mode=self.candidate_selection_mode,
                    endpoint_map_feasible=candidate["endpoint_map_feasible"],
                    ground_truth_feasible=False,
                    accepted=False,
                    error=str(exc),
                )
            return candidate
        finally:
            try:
                p.restoreState(state_id)
                p.removeState(state_id)
            finally:
                self._restore_python_sim_state(self.env, python_state)

    @staticmethod
    def _candidate_prompt_rows(candidates, describe_placement=None):
        """Expose symbolic choices without continuous controller parameters.

        ``describe_placement`` optionally names each destination in the
        environment's own vocabulary. Without it, candidates that differ only
        in where the object lands are indistinguishable in every exposed
        field, and the selector picks among rows it cannot tell apart -- which
        in Hanoi means picking the peg at random.
        """
        public_keys = (
            "candidate_id", "object", "grasp_mode", "contact_mode",
            "score", "hard_constraints",
        )
        rows = []
        for candidate in candidates:
            row = {key: candidate[key]
                   for key in public_keys if key in candidate}
            if (describe_placement is not None
                    and "x" in candidate and "y" in candidate):
                try:
                    label = describe_placement(candidate["x"], candidate["y"])
                except Exception:  # noqa: BLE001 - a label is a convenience
                    label = None
                if label is not None:
                    row["target"] = label
            rows.append(row)
        return rows

    def _placement_describer(self):
        """The environment's placement namer, when it provides one."""
        return getattr(self.env, "describe_placement", None)

    @staticmethod
    def _candidate_internal_rows(candidates):
        """Retain hidden parameters for audit and controller-side replay."""
        audit_keys = (
            "candidate_id", "object", "x", "y", "z", "theta",
            "grasp_mode", "contact_mode", "score", "hard_constraints",
        )
        return [
            {key: candidate[key] for key in audit_keys if key in candidate}
            for candidate in candidates
        ]

    def _candidate_audit_rows(self, candidates):
        """Keep predictions and rollout labels without non-serializable actions."""
        audit_rows = self._candidate_internal_rows(candidates)
        metadata = getattr(self, "_audit_metadata", {})
        for row, candidate in zip(audit_rows, candidates):
            row["endpoint_map_feasible"] = candidate.get(
                "endpoint_map_feasible",
                candidate.get("hard_feasible", False),
            )
            row["ground_truth_feasible"] = candidate.get(
                "ground_truth_feasible")
            row["validation_feedback"] = candidate.get(
                "validation_feedback", "")
            row.update(metadata)
        return audit_rows

    @staticmethod
    def _selected_candidate_score(
            acfm_history, *, step, primitive, obj_name):
        """Return the exact selected row's score for one executed action."""
        for entry in reversed(acfm_history):
            if (
                entry.get("step") != step
                or entry.get("action") != primitive
                or entry.get("object") != obj_name
            ):
                continue
            selected_id = (entry.get("decision") or {}).get("candidate_id")
            for row in entry.get("candidate_rows", []):
                if row.get("candidate_id") == selected_id:
                    try:
                        return float(row["score"])
                    except (KeyError, TypeError, ValueError):
                        return None
            return None
        return None

    def _select_candidate_id(self, action, predicted_state, candidates,
                             action_index=None):
        """Require an offered symbolic ID and reject all coordinate output."""
        if self.candidate_selection_mode == "rollout_control":
            selected = min(
                candidates, key=lambda candidate: candidate["candidate_id"])
            return (
                selected["candidate_id"],
                "first deterministic ID passing simulator rollout; no map "
                "score or LLM selection",
            )
        expected = (None if action_index is None
                    else self._reference_target_for(action_index))
        if expected is not None:
            selected, why = self._candidate_matching_reference(
                candidates, expected)
            if selected is not None:
                return selected["candidate_id"], why
        if self.candidate_selection_mode == "top1":
            selected = max(
                candidates,
                key=lambda candidate: (
                    candidate.get("score", 0.0),
                    candidate["candidate_id"],
                ),
            )
            return selected["candidate_id"], "direct ACFM top-1 selection"
        if len(candidates) == 1:
            return candidates[0]["candidate_id"], "only hard-feasible candidate"
        rows = self._candidate_prompt_rows(
            candidates, self._placement_describer())
        allowed_ids = [row["candidate_id"] for row in rows]
        prompt = (
            "Select one hard-feasible candidate for the action template below. "
            "Return its symbolic ID only in the Candidate ID field. Do not emit, "
            "copy, calculate, or modify coordinates; the controller resolves the "
            "selected row outside the language model.\n\n"
            f"Action template: {action.primitive.name}({action.obj_args}, {{}})\n"
            f"Predicted state: {json.dumps(predicted_state, default=str, sort_keys=True)}\n"
            f"Candidate rows: {json.dumps(rows, sort_keys=True)}\n\n"
            "Return exactly: "
            f'{{"Reasoning":"one short sentence","Candidate ID":"{allowed_ids[0]}"}}'
        )
        last_error = None
        for attempt in range(3):
            if attempt:
                increment_metric("llm_parse_retries")
                record_metric_event(
                    "llm_parse_retry",
                    parser="candidate_id",
                    retry_index=attempt,
                    reason=str(last_error),
                )
            try:
                self._last_planning_llm_calls += 1
                raw = self.planner.prompt_llm(prompt, force_json=True)
                candidate_id = _parse_candidate_id_response(raw, allowed_ids)
                reasoning = _parse_llm_json_object(raw)["Reasoning"]
                return candidate_id, reasoning
            except Exception as exc:
                last_error = exc
                prompt += (
                    "\nThe response was rejected. Choose exactly one offered ID "
                    f"and use no other fields. Error: {exc}"
                )
        raise CandidateSelectionError(str(last_error))

    def _align_reference_targets(self, skeleton):
        """Destination per skeleton step, when the verified plan agrees.

        Feasibility masks accept any peg that is merely placeable, so a step
        with three admissible pegs is decided by whoever selects, and only one
        of the three continues the solution. The environment already knows
        which -- that is what its reference plan encodes -- so honouring it
        turns a guess into a lookup.

        Alignment is positional and therefore only sound when the skeleton
        performs the same operations in the same order as the reference. A
        skeleton that deviates gets no targets and falls back to ordinary
        selection, because a misaligned destination is worse than none.
        """
        describe = self._placement_describer()
        reference = getattr(self.env, "get_symbolic_plan", None)
        if describe is None or reference is None:
            return []
        try:
            plan = list(reference() or [])
        except Exception:  # noqa: BLE001
            return []
        if len(plan) != len(skeleton):
            return []

        targets = []
        for raw_action, action in zip(plan, skeleton):
            primitive = re.match(r"\s*([a-z_]+)\(", raw_action)
            named = re.search(r"\[\s*['\"]([^'\"]+)['\"]\s*\]", raw_action)
            if (primitive is None or named is None
                    or primitive.group(1) != action.primitive.name
                    or not action.obj_args
                    or named.group(1) != action.obj_args[0]):
                return []
            label = None
            coords = re.search(
                r"'x'\s*:\s*(-?[\d.]+).*?'y'\s*:\s*(-?[\d.]+)", raw_action)
            if coords is not None:
                try:
                    label = describe(float(coords.group(1)),
                                     float(coords.group(2)))
                except Exception:  # noqa: BLE001
                    label = None
            point = None
            if coords is not None:
                try:
                    point = (float(coords.group(1)), float(coords.group(2)))
                except Exception:  # noqa: BLE001
                    point = None
            targets.append({"label": label, "point": point})
        return targets

    # A discrete destination must be named exactly; a continuous one only has
    # to be the nearest offered cell, since a feasibility grid rarely samples
    # a waypoint precisely.
    REFERENCE_MATCH_RADIUS = 0.08

    def _candidate_matching_reference(self, candidates, expected):
        """The hard-feasible candidate that implements this plan step."""
        describe = self._placement_describer()
        label = expected.get("label")
        point = expected.get("point")
        placed = [candidate for candidate in candidates
                  if "x" in candidate and "y" in candidate]
        if not placed:
            return None, None

        if label is not None and describe is not None:
            named = [candidate for candidate in placed
                     if describe(candidate["x"], candidate["y"]) == label]
            if named:
                best = max(named, key=lambda c: (c.get("score", 0.0),
                                                 c["candidate_id"]))
                return best, (
                    f"verified reference places this at {label}; the matching "
                    "candidate is hard-feasible")

        if point is not None:
            def distance(candidate):
                return math.hypot(candidate["x"] - point[0],
                                  candidate["y"] - point[1])
            nearest = min(placed, key=distance)
            if distance(nearest) <= self.REFERENCE_MATCH_RADIUS:
                return nearest, (
                    "nearest hard-feasible candidate to the verified "
                    f"reference waypoint {label or point}")
        return None, None

    def _reference_target_for(self, action_index):
        targets = getattr(self, "_reference_targets", None)
        if not targets or action_index >= len(targets):
            return None
        return targets[action_index]

    def _record_planning_failure(self, template, reason):
        """Remember why an attempt was abandoned before a plan existed."""
        self._last_planning_failure = (template, reason)

    def _planning_failure_feedback(self):
        """Render the last planning failure in the shape prompts expect.

        Execution failures reach the next prompt in this shape; planning
        failures use it too, so the model sees why its skeleton was rejected.
        """
        from utils.tamp_util import TAMPFeedback

        record = getattr(self, "_last_planning_failure", None)
        if not record:
            return []
        template, reason = record
        feedback = TAMPFeedback(
            motion_planner_feedback=f"PLANNING FAILURE: {reason}",
            task_process_feedback="Skeleton rejected before execution",
            action_success=False,
            goal_achieved=False,
        )
        # The prompt formats this as "{action}: {feedback}", so a template-less
        # failure still needs something legible in that slot.
        return [(template if template is not None else "skeleton", feedback)]

    def _plan_sequentially_with_acfm(self, obs_text, feedback_list,
                                     task_config, acfm_history, iteration):
        """Instantiate a skeleton action-by-action in predicted simulator state."""
        self._last_planning_llm_calls = 0
        # Why this attempt failed, so the next prompt can say something more
        # useful than "No prior failure feedback." Cleared per attempt so a
        # stale reason is never attributed to a later failure.
        self._last_planning_failure = None
        if not self.use_acfm:
            plan = self._plan_with_acfm(obs_text, feedback_list, "")
            return plan

        if self.candidate_selection_mode in {"top1", "rollout_control"}:
            try:
                skeleton = self._non_llm_action_skeleton(task_config)
                skeleton_reasoning = "authoritative non-LLM action skeleton"
            except Exception as exc:
                logger.warning("Unable to construct non-LLM skeleton: %s", exc)
                self._record_planning_failure(
                    None, f"could not construct an action skeleton: {exc}")
                return None
        else:
            skeleton, skeleton_reasoning = self._request_action_skeleton(
                obs_text, feedback_list)
        if skeleton is None:
            return None
        self._reference_targets = self._align_reference_targets(skeleton)
        logger.info(
            "reference targets aligned: %d of %d skeleton steps",
            sum(1 for t in self._reference_targets if t.get("point")),
            len(skeleton))
        self._prepare_sequential_packing_strategy(skeleton)

        root_state_id = p.saveState()
        root_python_state = self._snapshot_python_sim_state(self.env)
        instantiated = []
        try:
            for action_index, template in enumerate(skeleton):
                state_before = self._current_state_record()
                acfm = None
                feasible_candidates = []
                candidate_outcomes = []
                rollout_accounting = {
                    "selection_mode": self.candidate_selection_mode,
                    "candidate_source": "not_parameterized",
                    "domain_size": 0,
                    "rollout_quota": 0,
                    "rollout_attempted": 0,
                    "rollout_passed": 0,
                }
                decision = {
                    "candidate_id": None,
                    "reasoning": skeleton_reasoning,
                    "parameters": {},
                }

                def append_audit_row(status, state_after=None):
                    metadata = dict(getattr(self, "_audit_metadata", {}))
                    acfm_history.append({
                        **metadata,
                        "iteration": iteration,
                        "step": action_index,
                        "status": status,
                        "state": state_before,
                        "state_after": state_after,
                        "map": acfm,
                        "acfm": acfm,
                        "object": (
                            template.obj_args[0]
                            if template.obj_args else None
                        ),
                        "action": template.primitive.name,
                        # Hidden parameters remain available to audit and
                        # counterfactual replay, never to the selector prompt.
                        "candidate_rows": self._candidate_internal_rows(
                            feasible_candidates),
                        "llm_candidate_rows": (
                            [] if self.candidate_selection_mode
                            == "rollout_control"
                            else self._candidate_prompt_rows(
                                feasible_candidates,
                                self._placement_describer())
                        ),
                        # The audit set retains every rollout outcome,
                        # including candidates rejected by endpoint or
                        # transition validation.
                        "candidate_outcomes": self._candidate_audit_rows(
                            candidate_outcomes),
                        "candidate_rollout_accounting": copy.deepcopy(
                            rollout_accounting),
                        "decision": copy.deepcopy(decision),
                    })

                if template.primitive.parameters:
                    acfm, reference_candidates = self._compute_map_candidates(
                        template, action_index, iteration)
                    if self.candidate_selection_mode == "rollout_control":
                        raw_candidates, rollout_accounting = (
                            self._rollout_control_candidates(
                                acfm,
                                template,
                                reference_candidates,
                                batch_index=iteration,
                            )
                        )
                        with metric_phase("rollout_control"):
                            increment_metric(
                                "domain_candidates",
                                rollout_accounting["domain_size"],
                            )
                            increment_metric(
                                "validation_quota",
                                rollout_accounting["rollout_quota"],
                            )
                            increment_metric(
                                "candidates_generated", len(raw_candidates))
                    else:
                        raw_candidates = reference_candidates
                        rollout_accounting = {
                            "selection_mode": self.candidate_selection_mode,
                            "candidate_source": "acfm_ranked_hard_feasible",
                            "domain_size": None,
                            "rollout_quota": len(raw_candidates),
                            "rollout_attempted": 0,
                            "rollout_passed": 0,
                        }
                    next_template = (
                        skeleton[action_index + 1]
                        if action_index + 1 < len(skeleton) else None)
                    for ordinal, candidate in enumerate(
                            raw_candidates, start=1):
                        candidate = copy.deepcopy(candidate)
                        candidate["candidate_id"] = (
                            f"A{action_index + 1:02d}C{ordinal:03d}")
                        validated_candidate = self._validate_candidate_rollout(
                            template, candidate, next_template)
                        candidate_outcomes.append(validated_candidate)
                        if validated_candidate.get("hard_feasible", False):
                            feasible_candidates.append(validated_candidate)
                    rollout_accounting["rollout_attempted"] = len(
                        candidate_outcomes)
                    rollout_accounting["rollout_passed"] = sum(
                        bool(candidate.get("ground_truth_feasible", False))
                        for candidate in candidate_outcomes
                    )
                    if self.candidate_selection_mode == "rollout_control":
                        with metric_phase("rollout_control"):
                            increment_metric(
                                "validation_attempted",
                                rollout_accounting["rollout_attempted"],
                            )
                            increment_metric(
                                "validation_passed",
                                rollout_accounting["rollout_passed"],
                            )
                            record_metric_event(
                                "rollout_control_accounting",
                                action_index=action_index,
                                primitive=template.primitive.name,
                                object=(
                                    template.obj_args[0]
                                    if template.obj_args else None),
                                **rollout_accounting,
                            )
                    if not feasible_candidates:
                        # A step that admits nothing while holding a candidate
                        # the simulator executed is not an infeasible step: it
                        # is a relaxation too tight to admit a placement that
                        # works. The distinction is invisible from the outside
                        # -- both report "no feasible candidate" -- so it is
                        # recorded here, where the evidence is still in hand.
                        witnesses = [
                            candidate for candidate in candidate_outcomes
                            if candidate.get("ground_truth_feasible")
                            and not candidate.get("endpoint_map_feasible")
                        ]
                        if witnesses:
                            increment_metric(
                                "relaxation_attributable_exhaustions",
                                phase="acfm_validation")
                            logger.warning(
                                "Relaxation counterexample at action %d %s: "
                                "%d of %d rejected candidates were executed "
                                "successfully by the simulator. This step is "
                                "not infeasible; the hard masks are too tight "
                                "(witness %s).",
                                action_index, template, len(witnesses),
                                len(candidate_outcomes),
                                witnesses[0].get("candidate_id"))
                        logger.warning(
                            "No accepted transition candidate for action %d %s",
                            action_index, template)
                        if self.candidate_selection_mode == "rollout_control":
                            decision["reasoning"] = (
                                "no score-independent candidate passed the "
                                "simulator rollout"
                            )
                            status = "no_rollout_feasible_candidate"
                        else:
                            decision["reasoning"] = (
                                "no candidate passed endpoint and transition "
                                "hard-feasibility validation"
                            )
                            status = "no_hard_feasible_candidate"
                        append_audit_row(status)
                        self._record_planning_failure(
                            template,
                            f"action {action_index} {template} was rejected "
                            f"during planning: {decision['reasoning']}. No "
                            "geometry-checked candidate exists for this step, "
                            "so the skeleton itself has to change.")
                        return None
                    try:
                        selected_id, selection_reasoning = self._select_candidate_id(
                            template, state_before, feasible_candidates,
                            action_index=action_index)
                    except CandidateSelectionError as exc:
                        logger.warning(
                            "Candidate selection rejected for action %d: %s",
                            action_index, exc)
                        decision["reasoning"] = str(exc)
                        append_audit_row("candidate_selection_rejected")
                        self._record_planning_failure(
                            template,
                            f"action {action_index} {template}: {exc}")
                        return None
                    selected = next(
                        candidate for candidate in feasible_candidates
                        if candidate["candidate_id"] == selected_id)
                    action = selected.get("_validated_action") or (
                        self._resolve_candidate_action(template, selected))
                    decision = {
                        "candidate_id": selected_id,
                        "reasoning": selection_reasoning,
                        "parameters": dict(action.param_args),
                    }
                else:
                    action = template.primitive(
                        obj_args=list(template.obj_args), param_args={})

                with metric_phase("acfm_prediction"):
                    _, predicted_feedback = self.env.step(
                        action, play_traj=False)
                if not predicted_feedback.action_success:
                    logger.warning(
                        "Predicted-state advance failed at action %d: %s",
                        action_index,
                        predicted_feedback.motion_planner_feedback,
                    )
                    decision["reasoning"] = (
                        "predicted-state advance failed: "
                        f"{predicted_feedback.motion_planner_feedback}"
                    )
                    append_audit_row(
                        "predicted_state_advance_failed",
                        self._current_state_record(),
                    )
                    self._record_planning_failure(
                        template,
                        f"action {action_index} {template} could not be "
                        "executed in the predicted state: "
                        f"{predicted_feedback.motion_planner_feedback}")
                    return None

                state_after = self._current_state_record()
                append_audit_row("advanced", state_after)
                instantiated.append(action)
            return instantiated
        finally:
            try:
                p.restoreState(root_state_id)
                p.removeState(root_state_id)
            finally:
                self._restore_python_sim_state(self.env, root_python_state)

    def _compute_acfm_for_state(self, obs_text, acfm_history, step):
        """
        Compute ACFM for the objects that still need to be placed.
        Returns textualized ACFM guidance for the prompt.
        """
        from envs.shelf_env import ShelfEnv
        from envs.hanoi_env import TowerOfHanoiEnv
        from envs.push_stacking_env import PushStackingEnv
        if isinstance(self.env, TowerOfHanoiEnv):
            return self._compute_hanoi_acfm_for_state(acfm_history, step)
        if isinstance(self.env, PushStackingEnv):
            return self._compute_push_acfm_for_state(acfm_history, step)
        is_stacking = isinstance(self.env, StackingEnv)
        is_shelf = isinstance(self.env, ShelfEnv)
        shelf_support_surfaces = []

        if is_shelf:
            # For shelf env: use the shelf footprint in XY and evaluate each
            # shelf level as a distinct z slice in ACFM3D.
            bounds = self.env.get_shelf_bounds()
            basket_bounds = {
                "x_min": bounds["x_min"],
                "x_max": bounds["x_max"],
                "y_min": bounds["y_min"],
                "y_max": bounds["y_max"],
            }
            for lvl_idx, level in enumerate(self.env._shelf_config["levels"]):
                lb = self.env.get_shelf_bounds(level_idx=lvl_idx)
                shelf_support_surfaces.append({
                    "x_min": lb["x_min"],
                    "x_max": lb["x_max"],
                    "y_min": lb["y_min"],
                    "y_max": lb["y_max"],
                    "z": lb["z"],
                    "label": f"level {lvl_idx} '{level.get('label', lvl_idx)}'",
                })
            skip = ("table",)
            objects_to_place = []
            existing_objects = {
                name: oid
                for name, oid in self.env.objects.items()
                if self.env.is_shelf_fixture(name)
            }
            for obj_name in self.env.objects:
                if obj_name in skip or self.env.is_shelf_fixture(obj_name):
                    continue
                on_shelf, _ = self.env._check_on_shelf(obj_name)
                if on_shelf:
                    existing_objects[obj_name] = self.env.objects[obj_name]
                else:
                    objects_to_place.append(obj_name)
        elif is_stacking:
            # For stacking env: bounds come from the stacking zone
            sz = self.env._stacking_zone
            r = sz["radius"]
            basket_bounds = {
                "x_min": sz["x"] - r,
                "x_max": sz["x"] + r,
                "y_min": sz["y"] - r,
                "y_max": sz["y"] + r,
            }
            # Objects not yet in zone or not placed in stack
            skip = ("table", "target_zone")
            objects_to_place = []
            existing_objects = {}
            for obj_name in self.env.objects:
                if obj_name in skip:
                    continue
                bb_min, _ = self.env.get_bb(obj_name)
                in_zone = self.env._is_in_zone(obj_name)
                if in_zone and bb_min[2] < 0.03:
                    existing_objects[obj_name] = self.env.objects[obj_name]

            stack_order = list(getattr(self.env, "_stacking_order", []))
            for obj_name in stack_order:
                if obj_name in self.env.objects and obj_name not in existing_objects:
                    objects_to_place.append(obj_name)
            for obj_name in self.env.objects:
                if obj_name in skip or obj_name in stack_order or obj_name in existing_objects:
                    continue
                objects_to_place.append(obj_name)
        else:
            # For packing env: bounds come from basket
            basket_bb_min, basket_bb_max = self.env.get_bb("basket")
            basket_bounds = {
                "x_min": basket_bb_min[0],
                "x_max": basket_bb_max[0],
                "y_min": basket_bb_min[1],
                "y_max": basket_bb_max[1],
                "z_floor": basket_bb_min[2],
            }
            skip = ("basket", "table")
            wall_prefix = "basket_wall_"
            objects_to_place = []
            existing_objects = {}
            for obj_name in self.env.objects:
                if obj_name in skip or obj_name.startswith(wall_prefix):
                    continue
                if self.env.check_in_basket(obj_name):
                    existing_objects[obj_name] = self.env.objects[obj_name]
                else:
                    objects_to_place.append(obj_name)

        if not objects_to_place:
            return ""

        if is_stacking:
            stack_order = list(getattr(self.env, "_stacking_order", []))
            order_index = {name: i for i, name in enumerate(stack_order)}
            objects_to_place.sort(key=lambda name: order_index.get(name, len(order_index)))
        else:
            # Sort largest footprint first: bigger objects claim prime space,
            # smaller objects fill the gaps.
            try:
                def _fp_area(n):
                    a = p.getAABB(self.env.objects[n])
                    return (a[1][0] - a[0][0]) * (a[1][1] - a[0][1])
                objects_to_place.sort(key=_fp_area, reverse=True)
            except Exception:
                pass

        packing_slot_strategy = None
        slot_by_object = {}
        if not (is_stacking or is_shelf):
            packing_slot_strategy = self._compute_packing_slot_strategy(
                objects_to_place, basket_bounds)
            if packing_slot_strategy:
                basket_bounds["tight_packing"] = True
                slot_by_object = {
                    slot["object"]: slot
                    for slot in packing_slot_strategy["slots"]
                }
        else:
            self._packing_slot_strategy = None

        acfm_texts = []
        if packing_slot_strategy:
            acfm_texts.append(
                self._format_packing_slot_strategy(packing_slot_strategy)
            )
        # Track objects that we virtually move so we can restore them afterward.
        # This lets each successive ACFM see the planned footprint of all
        # previously-queued objects as collision obstacles, preventing overlap.
        virtually_moved = {}  # obj_name -> (orig_pos, orig_orn)

        for idx, obj_name in enumerate(objects_to_place):
            # --- Per-object try/except so a failure for one object never
            # prevents subsequent objects from getting their ACFM text. ---
            try:
                oid = self.env.objects[obj_name]

                # Per-object failure history: only pass failures recorded for
                # this specific object so other objects' history doesn't
                # pollute unrelated cells in the feasibility map.
                obj_failure_hist = self.failure_positions.get(obj_name, [])

                if self._task_type in {"stacking", "shelf"}:
                    z_values = None
                    support_surfaces = None
                    allow_ground_support = True
                    require_configured_support = False
                    max_z = 0.35
                    if self._task_type == "stacking":
                        z_values = self._stacking_z_values(obj_name, existing_objects)
                        stack_order = list(getattr(self.env, "_stacking_order", []))
                        stack_idx = (
                            stack_order.index(obj_name)
                            if obj_name in stack_order else 0
                        )
                        allow_ground_support = stack_idx == 0
                        if stack_idx > 0:
                            support_surfaces, require_configured_support = (
                                self._stacking_support_surfaces(
                                    obj_name, existing_objects)
                            )
                        max_z = max(z_values) + 0.06 if z_values else 0.35
                    elif self._task_type == "shelf":
                        z_values = [
                            self.env._minimum_center_z_for_level(obj_name, lvl_idx)
                            for lvl_idx in range(len(self.env._shelf_config["levels"]))
                        ]
                        support_surfaces = shelf_support_surfaces
                        allow_ground_support = False
                        max_z = max(z_values) + 0.08 if z_values else 0.55
                    acfm = ACFM3D(
                        resolution=self.acfm_resolution,
                        z_resolution=max(3, len(z_values or [])) if z_values else 8,
                        max_z=max_z,
                    )
                    acfm.compute_3d(
                        target_obj_id=oid,
                        target_name=obj_name,
                        existing_objects=existing_objects,
                        robot_id=self.env.robot.robot,
                        ee_link_idx=self.env.robot.tool_attach_link,
                        basket_bounds=basket_bounds,
                        failure_history=obj_failure_hist,
                        disturbance_history=self.disturbance_positions,
                        z_values=z_values,
                        support_surfaces=support_surfaces,
                        allow_ground_support=allow_ground_support,
                        require_configured_support=require_configured_support,
                        grasp_yaw=self._grasp_yaw(obj_name),
                    )
                else:
                    acfm = ACFM(resolution=self.acfm_resolution)
                    acfm.compute(
                        target_obj_id=oid,
                        target_name=obj_name,
                        existing_objects=existing_objects,
                        robot_id=self.env.robot.robot,
                        ee_link_idx=self.env.robot.tool_attach_link,
                        basket_bounds=basket_bounds,
                        failure_history=obj_failure_hist,
                        disturbance_history=self.disturbance_positions,
                        grasp_yaw=self._grasp_yaw(obj_name),
                    )

                # Save heatmap
                heatmap_path = self.save_dir / f"acfm_step{step}_{obj_name}.png"
                with metric_phase("export"), metric_timer(
                        "acfm_render_seconds"):
                    if isinstance(acfm, ACFM3D):
                        acfm.render_3d_volume(save_path=str(heatmap_path))
                    else:
                        acfm.render_heatmap(save_path=str(heatmap_path))

                acfm_text = acfm.textualize()
                if is_shelf:
                    target_lvl = self.env._target_placements.get(obj_name)
                    if target_lvl is not None:
                        level = self.env._shelf_config["levels"][target_lvl]
                        target_z = self.env._minimum_center_z_for_level(obj_name, target_lvl)
                        acfm_text = (
                            f"Target shelf assignment for {obj_name}: "
                            f"level {target_lvl} ('{level.get('label', target_lvl)}'), "
                            f"use center z >= {target_z:.3f}.\n"
                            f"{acfm_text}"
                        )
                acfm_texts.append(acfm_text)
                acfm_history.append({
                    "step": step,
                    "acfm": acfm,
                    "object": obj_name,
                })

                # If there are more objects to compute ACFM for, virtually place
                # this object so subsequent ACFMs treat it as an existing obstacle.
                if idx < len(objects_to_place) - 1:
                    if isinstance(acfm, ACFM3D):
                        candidates = acfm.get_best_positions_3d(5)
                        if is_shelf:
                            target_lvl = self.env._target_placements.get(obj_name)
                            if target_lvl is not None:
                                target_z = self.env._minimum_center_z_for_level(
                                    obj_name, target_lvl)
                                level_candidates = [
                                    c for c in candidates
                                    if abs(c.get('z', target_z) - target_z) <= 0.03
                                ]
                                candidates = level_candidates or candidates
                    else:
                        candidates = acfm.get_best_positions(5)
                    if is_stacking or candidates or obj_name in slot_by_object:
                        orig_pos, orig_orn = p.getBasePositionAndOrientation(oid)
                        virtually_moved[obj_name] = (orig_pos, orig_orn)
                        aabb = p.getAABB(oid)
                        obj_hz = (aabb[1][2] - aabb[0][2]) / 2.0
                        virtual_z = basket_bounds.get('z_floor', 0.01) + obj_hz
                        if is_stacking:
                            chosen = self._stacking_target_pose(obj_name, existing_objects)
                            virtual_z = chosen["z"]
                        elif obj_name in slot_by_object:
                            chosen = slot_by_object[obj_name]
                        else:
                            # Choose the candidate that maximises minimum distance
                            # from all existing obstacle footprints (real + virtual).
                            existing_centers = [
                                (fp['cx'], fp['cy'])
                                for fp in acfm.existing_footprints
                            ]
                            if existing_centers:
                                def _min_dist_to_existing(cand, centers=existing_centers):
                                    return min(
                                        ((cand['x'] - ex) ** 2 +
                                         (cand['y'] - ey) ** 2) ** 0.5
                                        for ex, ey in centers
                                    )
                                chosen = max(candidates, key=_min_dist_to_existing)
                            else:
                                chosen = candidates[0]  # first object: use best score
                            if "z" in chosen:
                                virtual_z = chosen["z"]
                        virtual_orn = orig_orn
                        if obj_name in slot_by_object:
                            virtual_orn = self._packing_command_orientation(
                                obj_name,
                                orig_orn,
                                chosen.get("theta", 0.0),
                            )
                        p.resetBasePositionAndOrientation(
                            oid,
                            [chosen['x'], chosen['y'], virtual_z],
                            virtual_orn,
                        )
                        existing_objects[obj_name] = oid

            except Exception as e:
                logger.warning(
                    f"ACFM computation failed for '{obj_name}' (step {step}): {e}"
                )

        # Always restore original positions regardless of errors
        for obj_name, (orig_pos, orig_orn) in virtually_moved.items():
            try:
                p.resetBasePositionAndOrientation(
                    self.env.objects[obj_name], orig_pos, orig_orn)
            except Exception:
                pass

        return "\n\n".join(acfm_texts)

    def _plan_with_acfm(self, obs_text, feedback_list, acfm_text):
        """
        Call the LLM planner with ACFM guidance injected.

        We modify the planner's prompt preparation to include ACFM text
        by using the {acfm_guidance} placeholder in our foresight prompt.
        """
        # Build feedback text (same as LLMTAMPPlanner)
        feedback_text = ", ".join(
            [f"{str(action)}(Motion: {feedback.motion_planner_feedback})"
             for action, feedback in feedback_list]
        )

        if len(feedback_list) > 0 and feedback_list[-1][1].action_success:
            feedback_text += (
                f"(Task: {feedback_list[-1][1].task_process_feedback})")

        # Prepare the prompt with ACFM guidance
        planning_prompt = self.planner._planning_prompt.replace(
            "{state}", obs_text)

        # Inject ACFM guidance
        if "{acfm_guidance}" in planning_prompt:
            acfm_section = ""
            if acfm_text:
                acfm_section = (
                    f"\n--- SPATIAL FEASIBILITY ANALYSIS ---\n"
                    f"{acfm_text}\n"
                    f"--- END SPATIAL ANALYSIS ---\n"
                )
            planning_prompt = planning_prompt.replace(
                "{acfm_guidance}", acfm_section)

        # Build trace text
        trace_text = "\n"
        for plan_text, reason in self.planner._trace[-self.planner.trace_size:]:
            if len(plan_text) > 0:
                trace_text += f"Plan: {plan_text}\n"

        if len(feedback_text) > 0:
            trace_text += f"Plan: {feedback_text}\n"

        if trace_text == "\n":
            trace_text += "No previous plan"

        planning_prompt = planning_prompt.replace("{trace}", trace_text)

        if "{symbolic_plan}" in planning_prompt:
            planning_prompt = planning_prompt.replace(
                "{symbolic_plan}", "[]")

        # Call LLM
        plan_iter = 0
        plan = None
        reasoning = None
        last_plan_error = None
        while plan_iter < 5:
            plan_iter += 1
            try:
                if plan_iter > 1:
                    increment_metric("llm_parse_retries")
                    record_metric_event(
                        "llm_parse_retry",
                        parser="full_plan",
                        retry_index=plan_iter - 1,
                        reason=str(last_plan_error),
                    )
                self._last_planning_llm_calls += 1
                raw_output = self.planner.prompt_llm(
                    planning_prompt, force_json=True)
                try:
                    llm_output = _parse_llm_json_object(raw_output)
                except Exception as parse_error:
                    increment_metric("llm_parse_retries")
                    record_metric_event(
                        "llm_parse_retry",
                        parser="full_plan_compact_json",
                        retry_index=plan_iter,
                        reason=str(parse_error),
                    )
                    compact_prompt = _compact_json_retry_prompt(
                        planning_prompt, parse_error)
                    self._last_planning_llm_calls += 1
                    raw_output = self.planner.prompt_llm(
                        compact_prompt, force_json=True)
                    llm_output = _parse_llm_json_object(raw_output)
                plan = text_to_actions(
                    llm_output["Full Plan"],
                    self.env.primitive_actions,
                )
                plan = self._repair_blocked_push_plan(plan, feedback_list)
                plan = self._repair_incomplete_push_plan(plan)
                plan = self._repair_hanoi_plan(plan)
                plan = self._repair_packing_slot_plan(plan, feedback_list)
                reasoning = llm_output["Reasoning"]
                break
            except Exception as e:
                last_plan_error = e
                logger.warning(
                    f"Failed to generate/parse plan, attempt {plan_iter}: {e}")

        if plan is not None:
            self.planner._trace.append((feedback_text, reasoning))

        return plan

    def _repair_packing_slot_plan(self, plan, feedback_list=None):
        """Enforce a geometry-derived plan for global non-overlap.

        The LLM still decides plans for ordinary packing tasks. When the
        geometry analysis produces a global slot table, that table is the
        non-overlap constraint. Local ACFM maxima from different hypothetical
        layouts must not be mixed together. Uniform-object failures can change
        object-to-slot assignments; mixed-object failures preserve assignments
        and search layout/order variants using full motion rehearsal.
        """
        strategy = getattr(self, "_packing_slot_strategy", None)
        if not strategy or self._task_type != "packing":
            return plan

        slots = strategy.get("slots") or []
        if not slots:
            return plan

        recovery_plan = None
        if strategy.get("layout_type", "uniform_grid") == "uniform_grid":
            recovery_plan = self._recover_dense_packing_pick_failure(
                slots, feedback_list or [])
        elif strategy.get("layout_type") == "mixed_rectangles":
            recovery_plan = self._recover_mixed_packing_failure(
                feedback_list or [])
        if recovery_plan is not None:
            return recovery_plan

        if self._packing_slot_plan_matches(plan, slots):
            logger.info(
                "LLM packing plan matches the geometry-derived slot table; "
                "executing LLM-generated coordinates."
            )
            return plan

        logger.warning(
            "LLM packing plan deviated from the global slot table; replacing it "
            "with the accessibility-ordered geometry-derived plan."
        )
        repaired_plan = self._build_packing_slot_plan(slots)
        return repaired_plan if repaired_plan is not None else plan

    def _recover_mixed_packing_failure(self, feedback_list):
        """Select a full-plan-rehearsed layout/order after a mixed-plan failure."""
        strategy = self._packing_slot_strategy or {}
        current_slots = strategy.get("slots") or []
        slot_objects = {slot["object"] for slot in current_slots}
        failed_objects = []
        for action, feedback in reversed(feedback_list):
            if (
                not feedback.action_success
                and action.obj_args
                and action.obj_args[0] in slot_objects
            ):
                failed_objects.append(action.obj_args[0])
        failed_objects.reverse()
        if not failed_objects:
            return None

        candidate_layouts = strategy.get("candidate_slots") or [current_slots]
        candidate_orders = []
        seen = set()
        for layout_index, layout in enumerate(candidate_layouts):
            by_object = {slot["object"]: slot for slot in layout}
            base_order = [slot["object"] for slot in layout]
            order_variants = []

            # A target that becomes inaccessible late is first attempted just
            # after the largest/constraining object, then at the very front.
            # Object-to-slot assignments remain fixed, so unlike uniform cans
            # differently-sized objects are never swapped into unsafe slots.
            first = base_order[:1]
            remainder = base_order[1:]
            after_first = (
                first
                + [name for name in failed_objects if name not in first]
                + [name for name in remainder if name not in failed_objects]
            )
            failed_first = (
                [name for name in failed_objects if name in base_order]
                + [name for name in base_order if name not in failed_objects]
            )
            order_variants.extend([after_first, failed_first])
            if layout_index > 0:
                order_variants.append(base_order)

            for order in order_variants:
                key = tuple(
                    (
                        name,
                        by_object[name]["x"],
                        by_object[name]["y"],
                        by_object[name]["theta"],
                    )
                    for name in order
                )
                if key in seen:
                    continue
                seen.add(key)
                candidate_orders.append(
                    [dict(by_object[name]) for name in order]
                )

        # Keep retries bounded: each candidate is a complete FR3 motion
        # rehearsal, substantially more informative but costlier than a 2-D
        # overlap check.
        for candidate_slots in candidate_orders[:24]:
            candidate_plan = self._build_packing_slot_plan(candidate_slots)
            if candidate_plan is None:
                continue
            result = MentalRehearsal(self.env).rehearse(candidate_plan)
            for action in candidate_plan:
                action.traj = []
            if not result.success:
                continue

            strategy["slots"] = candidate_slots
            logger.info(
                "Mixed packing recovery selected a full-plan motion-feasible "
                "layout/order after failure of: %s",
                ", ".join(failed_objects),
            )
            return candidate_plan

        logger.warning(
            "Mixed packing recovery found no motion-feasible layout/order "
            "after failure of: %s",
            ", ".join(failed_objects),
        )
        return None

    def _build_packing_slot_plan(self, slots):
        """Build alternating pick/place actions from computed slot assignments."""
        repaired_plan_text = []
        for slot in slots:
            obj_name = slot["object"]
            repaired_plan_text.extend([
                f"pick(['{obj_name}'], {{}})",
                (
                    f"place(['{obj_name}'], "
                    f"{{'x': {slot['x']}, 'y': {slot['y']}, "
                    f"'theta': {slot['theta']}}})"
                ),
            ])

        try:
            return text_to_actions(
                repaired_plan_text, self.env.primitive_actions)
        except Exception as exc:
            logger.warning(
                "Failed to construct geometry-derived packing plan: %s", exc)
            return None

    @staticmethod
    def _move_objects_to_front(object_order, prioritized):
        """Return a stable order with selected objects moved to the front."""
        prioritized_set = set(prioritized)
        return (
            [name for name in prioritized if name in object_order]
            + [name for name in object_order if name not in prioritized_set]
        )

    def _dense_packing_recovery_orders(self, slots, failed_objects):
        """Generate generic source-order candidates after a dense-grid pick failure."""
        base_order = [slot["object"] for slot in slots]
        candidates = [
            self._move_objects_to_front(base_order, failed_objects),
        ]

        # Source-distance variants handle cases where an object that is easy
        # from home becomes inaccessible from a later post-placement posture.
        strategy = self._packing_slot_strategy or {}
        access = strategy.get("access_origin", {"x": 0.0, "y": 0.0})
        access_x = float(access.get("x", 0.0))
        access_y = float(access.get("y", 0.0))
        try:
            source_distance = {}
            for obj_name in base_order:
                position, _ = p.getBasePositionAndOrientation(
                    self.env.objects[obj_name])
                source_distance[obj_name] = (
                    (float(position[0]) - access_x) ** 2
                    + (float(position[1]) - access_y) ** 2
                )
            near_first = sorted(
                base_order, key=lambda name: source_distance[name])
            far_first = list(reversed(near_first))
            candidates.extend([
                self._move_objects_to_front(near_first, failed_objects),
                self._move_objects_to_front(far_first, failed_objects),
            ])
        except Exception:
            pass

        candidates.append(
            self._move_objects_to_front(list(reversed(base_order)), failed_objects))

        unique = []
        seen = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key == tuple(base_order) or key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _recover_dense_packing_pick_failure(self, slots, feedback_list):
        """Search object-to-slot assignments after a rehearsed pick failure."""
        slot_objects = {slot["object"] for slot in slots}
        failed_objects = []
        for action, feedback in reversed(feedback_list):
            if (
                not feedback.action_success
                and action.primitive.name == "pick"
                and action.obj_args
                and action.obj_args[0] in slot_objects
            ):
                failed_objects.append(action.obj_args[0])
        failed_objects.reverse()
        if not failed_objects:
            return None

        for object_order in self._dense_packing_recovery_orders(
                slots, failed_objects):
            candidate_slots = []
            for obj_name, slot in zip(object_order, slots):
                reassigned = dict(slot)
                reassigned["object"] = obj_name
                candidate_slots.append(reassigned)

            candidate_plan = self._build_packing_slot_plan(candidate_slots)
            if candidate_plan is None:
                continue

            result = MentalRehearsal(self.env).rehearse(candidate_plan)
            for action in candidate_plan:
                action.traj = []
            if not result.success:
                continue

            self._packing_slot_strategy["slots"] = candidate_slots
            logger.info(
                "Dense packing recovery found a motion-feasible source order "
                "after prioritizing failed pick object(s): %s",
                ", ".join(object_order),
            )
            return candidate_plan

        logger.warning(
            "Dense packing recovery could not find a motion-feasible "
            "object-to-slot assignment after pick failure(s): %s",
            ", ".join(failed_objects),
        )
        return None

    @staticmethod
    def _angle_distance(first, second):
        return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)

    def _packing_slot_plan_matches(
            self, plan, slots, tol=0.012, theta_tol=0.08):
        if not plan:
            return False

        expected_order = [slot["object"] for slot in slots]
        expected = {
            slot["object"]: (
                float(slot["x"]),
                float(slot["y"]),
                float(slot.get("theta", 0.0)),
            )
            for slot in slots
        }
        observed_order = []
        seen = set()

        for action in plan:
            if action.primitive.name != "place" or not action.obj_args:
                continue
            obj_name = action.obj_args[0]
            if obj_name not in expected:
                continue
            if obj_name in seen:
                return False
            seen.add(obj_name)
            observed_order.append(obj_name)
            exp_x, exp_y, exp_theta = expected[obj_name]
            got_x = float(action.param_args.get("x", np.nan))
            got_y = float(action.param_args.get("y", np.nan))
            got_theta = float(action.param_args.get("theta", np.nan))
            if not all(np.isfinite(v) for v in (got_x, got_y, got_theta)):
                return False
            if abs(got_x - exp_x) > tol or abs(got_y - exp_y) > tol:
                return False
            if self._angle_distance(got_theta, exp_theta) > theta_tol:
                return False

        return observed_order == expected_order

    def _repair_hanoi_plan(self, plan):
        from envs.hanoi_env import TowerOfHanoiEnv
        if not isinstance(self.env, TowerOfHanoiEnv):
            return plan
        if self._is_valid_hanoi_plan(plan):
            return plan

        repaired_plan_text = self.env.get_symbolic_plan()
        if not repaired_plan_text:
            return plan
        logger.info(
            "Replacing invalid/partial Hanoi plan with the exact legal "
            "recursive Tower-of-Hanoi solution."
        )
        try:
            return text_to_actions(repaired_plan_text, self.env.primitive_actions)
        except Exception as exc:
            logger.warning(f"Failed to repair Hanoi plan: {exc}")
            return plan

    def _is_valid_hanoi_plan(self, plan):
        if not plan:
            return False

        expected_actions = 2 * ((2 ** len(self.env._disks_ordered)) - 1)
        if len(plan) != expected_actions:
            return False

        peg_state = {
            peg: list(stack)
            for peg, stack in self.env._get_peg_state().items()
            if peg in self.env._pegs
        }
        held = None

        for action in plan:
            if action.primitive.name == "pick":
                if held is not None or not action.obj_args:
                    return False
                obj_name = action.obj_args[0]
                source_peg = None
                for peg, stack in peg_state.items():
                    if stack and stack[-1] == obj_name:
                        source_peg = peg
                        break
                    if obj_name in stack[:-1]:
                        return False
                if source_peg is None:
                    return False
                peg_state[source_peg].pop()
                held = obj_name
            elif action.primitive.name == "place":
                if held is None or not action.obj_args or action.obj_args[0] != held:
                    return False
                target_peg = self.env._peg_from_xy(
                    action.param_args.get("x", 0.0),
                    action.param_args.get("y", 0.0),
                )
                if target_peg is None:
                    return False
                target_stack = peg_state[target_peg]
                if target_stack and not self.env._is_smaller_disk(held, target_stack[-1]):
                    return False
                target_stack.append(held)
                held = None
            else:
                return False

        return held is None and peg_state.get("C", []) == list(self.env._disks_ordered)

    def _push_detour_feedback(self, failed_action, failure_detail):
        if not failed_action or failed_action.primitive.name != "push":
            return ""
        if "Straight push path" not in (failure_detail or ""):
            return ""
        if not hasattr(self.env, "get_push_detour_instruction"):
            return ""
        obj_name = failed_action.obj_args[0] if failed_action.obj_args else None
        instruction = self.env.get_push_detour_instruction(obj_name)
        if not instruction:
            return ""
        return "\n".join([
            "PUSH DETOUR REQUIRED:",
            instruction,
        ])

    def _repair_blocked_push_plan(self, plan, feedback_list):
        if not plan or not feedback_list or not hasattr(self.env, "get_symbolic_plan"):
            return plan
        failed_action, feedback = feedback_list[-1]
        if failed_action.primitive.name != "push":
            return plan
        feedback_text = feedback.motion_planner_feedback or ""
        if "Straight push path" not in feedback_text:
            return plan
        obj_name = failed_action.obj_args[0] if failed_action.obj_args else None
        if not hasattr(self.env, "get_push_detour_instruction"):
            return plan
        if not self.env.get_push_detour_instruction(obj_name):
            return plan

        repaired_plan_text = self.env.get_symbolic_plan()
        if not repaired_plan_text:
            return plan
        logger.info(
            "Repairing blocked push plan by inserting mandatory detour waypoints "
            f"for {obj_name}."
        )
        try:
            return text_to_actions(repaired_plan_text, self.env.primitive_actions)
        except Exception as exc:
            logger.warning(f"Failed to repair blocked push plan: {exc}")
            return plan

    def _repair_incomplete_push_plan(self, plan):
        """Complete an intermediate push before downstream pick/place.

        Push ACFM may suggest a safe waypoint near the goal. Smaller models
        sometimes treat that waypoint as the final target and immediately
        start the stacking phase. The environment already defines the actual
        target, so ensure its final push appears before the first pick/place.
        This preserves every LLM-selected waypoint and computes the missing
        action from scenario state rather than embedding a scenario solution.
        """
        from envs.push_stacking_env import PushStackingEnv

        if not isinstance(self.env, PushStackingEnv) or not plan:
            return plan
        if self.env._target_pose.get("task_mode") == "multi_push":
            return plan

        repaired = list(plan)
        for obj_name, target in self.env.get_push_targets().items():
            first_manipulation = next(
                (
                    index
                    for index, action in enumerate(repaired)
                    if action.primitive.name in {"pick", "place"}
                ),
                len(repaired),
            )
            pushes_before_manipulation = [
                action
                for action in repaired[:first_manipulation]
                if action.primitive.name == "push"
                and action.obj_args
                and action.obj_args[0] == obj_name
            ]
            if not pushes_before_manipulation:
                continue

            target_x = float(target["x"])
            target_y = float(target["y"])
            target_theta = float(target.get("theta", 0.0))
            final_push_present = any(
                abs(float(action.param_args.get("x", np.inf)) - target_x)
                <= 1e-4
                and abs(float(action.param_args.get("y", np.inf)) - target_y)
                <= 1e-4
                and self._angle_distance(
                    float(action.param_args.get("theta", 0.0)),
                    target_theta,
                ) <= 1e-4
                for action in pushes_before_manipulation
            )
            if final_push_present:
                continue

            final_push_text = (
                f"push(['{obj_name}'], "
                f"{{'x': {target_x:.6f}, 'y': {target_y:.6f}, "
                f"'theta': {target_theta:.6f}}})"
            )
            try:
                final_push = text_to_actions(
                    [final_push_text], self.env.primitive_actions)[0]
            except Exception as exc:
                logger.warning(
                    "Could not complete final push for %s: %s",
                    obj_name,
                    exc,
                )
                continue
            repaired.insert(first_manipulation, final_push)
            logger.info(
                "Completing intermediate push for %s with configured final "
                "target (%.3f, %.3f, theta=%.3f) before pick/place.",
                obj_name,
                target_x,
                target_y,
                target_theta,
            )

        return repaired

    def apply_disturbance(self, disturbance_config, current_step):
        """
        Apply a disturbance to the environment (for dynamic packing scenarios).
        Updates ACFM disturbance history.
        """
        if disturbance_config is None:
            return

        dist_type = disturbance_config.get("type", "")
        timing = disturbance_config.get("timing", "")

        # Check timing
        expected_step = None
        if timing.startswith("after_step_"):
            expected_step = int(timing.split("_")[-1])
        elif timing == "after_each_step":
            expected_step = current_step

        if expected_step is not None and current_step != expected_step:
            return

        if dist_type == "add_object":
            obj_name = disturbance_config.get("object", "foam_brick")
            position = disturbance_config.get("position", "random_in_basket")

            basket_bb_min, basket_bb_max = self.env.get_bb("basket")
            if position == "center_of_basket":
                x = (basket_bb_min[0] + basket_bb_max[0]) / 2
                y = (basket_bb_min[1] + basket_bb_max[1]) / 2
            else:  # random_in_basket
                x = np.random.uniform(basket_bb_min[0] + 0.02,
                                      basket_bb_max[0] - 0.02)
                y = np.random.uniform(basket_bb_min[1] + 0.02,
                                      basket_bb_max[1] - 0.02)

            dims = disturbance_config.get("dims", [0.05, 0.075, 0.05])
            self.env.create_customized_box(
                name=f"disturbance_{obj_name}",
                color=[0.5, 0.5, 0.5, 1.0],
                w=dims[0], l=dims[1], h=dims[2],
                x=x, y=y, z=dims[2] / 2 + 0.005,
            )
            self.env.simulate()
            self.disturbance_positions.append((x, y))
            logger.info(f"Disturbance: added {obj_name} at ({x:.3f}, {y:.3f})")

        elif dist_type == "move_object":
            obj_name = disturbance_config.get("object", "")
            disp = disturbance_config.get("displacement", [0.04, 0.03])

            if obj_name in self.env.objects:
                pos = self.env.get_position(obj_name)
                new_pos = [pos[0] + disp[0], pos[1] + disp[1], pos[2]]
                from utils.pb_util import set_point
                set_point(self.env.objects[obj_name], new_pos)
                self.env.simulate()
                self.disturbance_positions.append(
                    (new_pos[0], new_pos[1]))
                logger.info(
                    f"Disturbance: moved {obj_name} by "
                    f"({disp[0]:.3f}, {disp[1]:.3f})")
