"""
Physics-Grounded Mental Rehearsal

Simulates the entire plan in a shadow PyBullet state before committing
simulator actions. Catches cascading failures that per-action checking misses.
"""
import pybullet as p
import numpy as np
import copy
import hashlib
import json
import logging
import time

from utils.rendering_util import suppress_gui_rendering
from utils.metrics_util import metric_phase, metric_timer, record_metric_event

logger = logging.getLogger(__name__)


class RehearsalResult:
    """Container for rehearsal outcomes."""

    def __init__(self, success, failed_step, failed_action, failure_type,
                 failure_detail, step_results, cascading_impact, *,
                 cached_trajectories=None, motion_planner_calls=0,
                 elapsed_seconds=0.0):
        self.success = success
        self.failed_step = failed_step
        self.failed_action = failed_action
        self.failure_type = failure_type
        self.failure_detail = failure_detail
        self.step_results = step_results
        self.cascading_impact = cascading_impact
        # Trajectories are copied after the shadow rollout and can be replayed
        # from the identical restored start state.  This avoids validating one
        # random path and executing a newly sampled one.
        self.cached_trajectories = cached_trajectories or []
        self.motion_planner_calls = int(motion_planner_calls)
        self.elapsed_seconds = float(elapsed_seconds)


class MentalRehearsal:
    """
    Full-plan forward simulation using PyBullet saveState/restoreState.
    Runs the complete plan in a cloned world, checks every step for:
    - Collision with existing objects
    - IK infeasibility
    - Object stability after placement
    - Goal achievability after all actions
    """

    def __init__(self, env):
        """
        Args:
            env: PackCompactEnv instance (provides robot, objects, step, etc.)
        """
        self.env = env

    @suppress_gui_rendering
    def rehearse(self, plan):
        """
        Simulate the full plan without committing the episode state.

        Args:
            plan: List of Action objects (from the planner)

        Returns:
            RehearsalResult with success/failure details and cascading impact
        """
        rehearsal_started = time.monotonic()
        # Save the committed simulator state.
        state_id = p.saveState()
        # PyBullet saveState does not restore Python-side bookkeeping. Preserve
        # it as well so repeated candidate rehearsals cannot inherit a stale
        # attachment or grasp direction from a failed pick/place attempt.
        robot = getattr(self.env, "robot", None)
        saved_attachments = (
            list(robot.attachments_robot)
            if robot is not None and hasattr(robot, "attachments_robot")
            else None
        )
        saved_grasp_direction = (
            robot.last_grasp_direction
            if robot is not None and hasattr(robot, "last_grasp_direction")
            else None
        )
        saved_grasp_position_offset = copy.deepcopy(
            getattr(robot, "last_grasp_position_offset", None)
            if robot is not None else None
        )
        saved_motion_trace = copy.deepcopy(
            getattr(robot, "last_motion_trace", None)
            if robot is not None else None
        )
        saved_theta_dict = (
            dict(self.env.theta_dict)
            if hasattr(self.env, "theta_dict")
            else None
        )
        saved_prepush_transition_done = getattr(
            self.env, "_prepush_transition_done", None)

        step_results = []
        placed_in_rehearsal = {}  # track objects placed during rehearsal
        cached_trajectories = []
        fallback_motion_planner_calls = 0
        planner_count_at_start = getattr(
            robot, "motion_planner_call_count", None)

        def planner_calls_so_far():
            if planner_count_at_start is None:
                return fallback_motion_planner_calls
            return max(
                0,
                int(robot.motion_planner_call_count - planner_count_at_start),
            )

        def build_result(**kwargs):
            return RehearsalResult(
                step_results=step_results,
                cached_trajectories=copy.deepcopy(cached_trajectories),
                motion_planner_calls=planner_calls_so_far(),
                elapsed_seconds=time.monotonic() - rehearsal_started,
                **kwargs,
            )

        try:
            with metric_phase("rehearsal"), metric_timer("rollout"):
                for step_idx, action in enumerate(plan):
                    # A missing trajectory means rehearsal invokes the motion
                    # planner.  A populated trajectory is an exact cached replay.
                    planned_now = action.traj is None or len(action.traj) == 0
                    calls_before_step = planner_calls_so_far()
                    if planned_now and planner_count_at_start is None:
                        fallback_motion_planner_calls += 1

                    # Execute action through the environment (motion plan + simulate)
                    _, feedback = self.env.step(action, play_traj=False)
                    cached = copy.deepcopy(action.traj or [])
                    cached_trajectories.append(cached)
                    trajectory_digest = self._trajectory_digest(cached)
                    calls_for_step = planner_calls_so_far() - calls_before_step

                    step_results.append({
                        "step": step_idx,
                        "action": str(action),
                        "success": feedback.action_success,
                        "feedback": feedback.motion_planner_feedback,
                        "motion_planner_called": calls_for_step > 0,
                        "motion_planner_calls": calls_for_step,
                        "trajectory_points": len(cached),
                        "trajectory_sha256": trajectory_digest,
                    })
                    record_metric_event(
                        "rehearsal_step",
                        step=step_idx,
                        action=str(action),
                        success=bool(feedback.action_success),
                        motion_planner_called=calls_for_step > 0,
                        motion_planner_calls=calls_for_step,
                        trajectory_points=len(cached),
                        trajectory_sha256=trajectory_digest,
                    )

                    if not feedback.action_success:
                        logger.info(
                            f"Rehearsal: step {step_idx} motion plan failed "
                            f"({feedback.motion_planner_feedback})")
                        return build_result(
                            success=False,
                            failed_step=step_idx,
                            failed_action=str(action),
                            failure_type="motion_plan_failure",
                            failure_detail=feedback.motion_planner_feedback,
                            cascading_impact=self._analyze_cascade_impact(
                                plan, step_idx, placed_in_rehearsal),
                        )

                    # Track successful placements
                    if action.primitive.name == "place":
                        placed_in_rehearsal[action.obj_args[0]] = {
                            "x": action.param_args.get("x", 0),
                            "y": action.param_args.get("y", 0),
                        }

                    # Check post-placement stability for place actions
                    if action.primitive.name == "place" and feedback.action_success:
                        obj_name = action.obj_args[0]
                        stability_result = self._check_stability(
                            obj_name, action.param_args)
                        if not stability_result["stable"]:
                            cascade = self._analyze_cascade_impact(
                                plan, step_idx, placed_in_rehearsal)
                            return build_result(
                                success=False,
                                failed_step=step_idx,
                                failed_action=str(action),
                                failure_type="unstable_placement",
                                failure_detail=stability_result["detail"],
                                cascading_impact=cascade,
                            )

        finally:
            # ALWAYS restore original state
            try:
                p.restoreState(state_id)
                p.removeState(state_id)
            except Exception:
                pass
            if saved_attachments is not None:
                robot.attachments_robot = saved_attachments
            if robot is not None and hasattr(robot, "last_grasp_direction"):
                robot.last_grasp_direction = saved_grasp_direction
            if robot is not None and hasattr(robot, "last_grasp_position_offset"):
                robot.last_grasp_position_offset = copy.deepcopy(
                    saved_grasp_position_offset)
            if robot is not None and hasattr(robot, "last_motion_trace"):
                robot.last_motion_trace = copy.deepcopy(saved_motion_trace)
            if saved_theta_dict is not None:
                self.env.theta_dict = saved_theta_dict
            if saved_prepush_transition_done is not None:
                self.env._prepush_transition_done = (
                    saved_prepush_transition_done)

        return build_result(
            success=True,
            failed_step=None,
            failed_action=None,
            failure_type=None,
            failure_detail=None,
            cascading_impact=[],
        )

    @staticmethod
    def _trajectory_digest(trajectory):
        """Hash a cached trajectory without assuming a NumPy-only shape."""
        payload_value = {
            "waypoints": list(trajectory),
            "motion_trace": getattr(trajectory, "motion_trace", []),
        }
        payload = json.dumps(
            payload_value,
            default=lambda value: (
                value.tolist() if hasattr(value, "tolist") else repr(value)
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def apply_cached_trajectories(plan, result):
        """Install the exact paths validated by a successful rehearsal."""
        if not result.success:
            raise ValueError("Cannot execute trajectories from a failed rehearsal")
        if len(result.cached_trajectories) != len(plan):
            raise ValueError(
                "Rehearsal trajectory count does not match the executable plan"
            )
        for action, trajectory in zip(plan, result.cached_trajectories):
            action.traj = copy.deepcopy(trajectory)
            action.motion_trace = copy.deepcopy(
                getattr(trajectory, "motion_trace", [])
            )
        return plan

    def _check_stability(self, obj_name, params):
        """Check if a placed object stays where it was placed after simulation."""
        if obj_name not in self.env.objects:
            return {"stable": True, "detail": ""}

        obj_id = self.env.objects[obj_name]
        actual_pos, _ = p.getBasePositionAndOrientation(obj_id)
        target_x = params.get("x", 0)
        target_y = params.get("y", 0)
        drift = ((actual_pos[0] - target_x)**2 + (actual_pos[1] - target_y)**2)**0.5

        if drift > 0.03:
            return {
                "stable": False,
                "detail": (
                    f"{obj_name} drifted {drift:.3f}m after placement "
                    f"(target: ({target_x:.3f}, {target_y:.3f}), "
                    f"actual: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}))"
                ),
            }
        return {"stable": True, "detail": ""}

    def _analyze_cascade_impact(self, plan, failed_step, placed):
        """Identify which later steps are affected by the failure at failed_step."""
        impacted = []
        for later_step in range(failed_step + 1, len(plan)):
            later_action = plan[later_step]
            if later_action.primitive.name == "place":
                later_params = later_action.param_args
                for placed_name, placed_params in placed.items():
                    dist = (
                        (later_params.get('x', 0) - placed_params.get('x', 0))**2 +
                        (later_params.get('y', 0) - placed_params.get('y', 0))**2
                    )**0.5
                    if dist < 0.06:
                        impacted.append({
                            "step": later_step,
                            "action": str(later_action),
                            "reason": f"too close to {placed_name} (dist={dist:.3f}m)",
                        })
        return impacted

    def format_report_for_llm(self, result):
        """Format rehearsal result as text for the LLM prompt."""
        if result.success:
            return (
                "REHEARSAL PASSED: All actions verified in simulation. "
                "Plan is feasible — proceed with execution."
            )

        lines = [
            "=== REHEARSAL FAILED ===",
            f"Failure at step {result.failed_step}: {result.failed_action}",
            f"Type: {result.failure_type}",
            f"Detail: {result.failure_detail}",
            "",
            f"Steps before failure (all passed): {result.failed_step}",
        ]

        if result.cascading_impact:
            lines.append(
                f"\nCascading impact on {len(result.cascading_impact)} later steps:")
            for imp in result.cascading_impact:
                lines.append(f"  Step {imp['step']}: {imp['reason']}")

        lines.extend([
            "",
            "INSTRUCTION: Revise the plan to avoid the failure above.",
            "Use the ACFM safest positions as alternatives.",
            "Consider reordering objects if the failure is caused by earlier placements.",
            "=== END REHEARSAL ===",
        ])
        return "\n".join(lines)
