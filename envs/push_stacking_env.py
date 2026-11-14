import logging
import math
import numpy as np

import pybullet as p
from pybullet_planning import (
    get_joint_positions,
    get_link_pose,
    set_joint_positions,
    wait_for_duration,
    LockRenderer,
)
from pybullet_planning.interfaces.robots.collision import get_collision_fn

from envs.pb_env import PybulletEnv
from utils import replay_capture
from utils.dynamics_util import apply_configured_dynamics
from utils.llm_util import textualize_array
from utils.metrics_util import increment_metric
from utils.pb_util import create_box, set_point
from utils.tamp_util import Action, Parameter, PrimitiveAction

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Feedback emitted when a push target is already satisfied within tolerance.
PUSH_ALREADY_AT_TARGET = "Already at push target."

class PushStackingEnv(PybulletEnv):
    """
    Push-then-stack environment.

    A large base object is intentionally too wide to grasp. The robot must push
    it into a target zone, then pick a smaller object and place it on top.
    """

    def __init__(self):
        super().__init__()
        self._primitive_actions = {
            "push": PrimitiveAction(
                name="push",
                obj_arity=1,
                parameters={
                    "x": Parameter("x", lower_limit=0.0, upper_limit=0.70),
                    "y": Parameter("y", lower_limit=-0.615, upper_limit=0.615),
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
            "pick": PrimitiveAction(name="pick", obj_arity=1),
            "place": PrimitiveAction(
                name="place",
                obj_arity=1,
                parameters={
                    "x": Parameter("x", lower_limit=0.0, upper_limit=0.70),
                    "y": Parameter("y", lower_limit=-0.615, upper_limit=0.615),
                    "z": Parameter("z", lower_limit=0.0, upper_limit=0.6),
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
        }
        self._objects_cfg = {}
        self._stacking_order = []
        self._fixture_names = set()
        self._object_targets = {}
        self._target_pose = {
            "x": 0.58,
            "y": -0.05,
            "theta": 0.0,
            "radius": 0.05,
            "push_force_max": 28.0,
            "push_force_min": 0.4,
            "push_force_kp": 140.0,
            "push_force_kd": 18.0,
            "push_end_margin": 0.0,
            "push_contact_lateral_gain": 0.5,
            "push_contact_resolution": 0.00025,
            "push_lost_contact_grace": 6,
            "push_ee_z_above_contact": 0.035,
        }
        self._graspable = {}
        self._max_grasp_width = 0.08
        self._initial_joint_positions = None
        self._approach_tcp_pose = None
        self._prepush_transition_done = False

    def __str__(self):
        return "PushStackingEnv"

    def reset(self, boxes, stacking_order, target_pose=None, use_gui=True, **kwargs):
        # The shared reset provides the same texture-free floor and linked
        # studio backdrop used by every demo.
        super().reset(use_gui=use_gui)
        self._objects_cfg = boxes
        self._stacking_order = stacking_order
        self._target_pose = target_pose or self._target_pose
        self._graspable = {}
        self._fixture_names = {"table", "target_zone"}
        self._max_grasp_width = kwargs.get("max_grasp_width", 0.08)
        self._initial_joint_positions = kwargs.get("initial_joint_positions")
        self._approach_tcp_pose = kwargs.get("approach_tcp_pose")
        self._prepush_transition_done = False
        self._object_targets = self._normalize_push_targets()

        self.create_table(color=[0.82, 0.82, 0.82, 1.0])

        for idx, (target_name, tp) in enumerate(self._object_targets.items()):
            marker_h = tp.get("marker_h", self._target_pose.get("marker_h", 0.002))
            marker_z = tp.get(
                "marker_z",
                self._target_pose.get(
                    "marker_z",
                    self.get_push_bounds()["z_floor"] + marker_h / 2.0,
                ),
            )
            marker = create_box(
                w=tp.get("marker_w", self._target_pose.get("marker_w", 0.22)),
                l=tp.get("marker_l", self._target_pose.get("marker_l", 0.16)),
                h=marker_h,
                mass=0,
                color=tp.get("marker_color", [0.2, 0.8, 0.55, 0.35]),
            )
            marker_name = "target_zone" if idx == 0 else f"target_zone_{target_name}"
            set_point(marker, [tp["x"], tp["y"], marker_z])
            p.changeDynamics(marker, -1, mass=0)
            if not tp.get("marker_collidable", self._target_pose.get("marker_collidable", False)):
                p.setCollisionFilterGroupMask(marker, -1, 0, 0)
            self._objects[marker_name] = marker
            self._fixture_names.add(marker_name)

        for box_info in boxes.values():
            name = box_info["name"]
            w = box_info["w"]
            l = box_info["l"]
            h = box_info["h"]
            roll = box_info.get("roll", 0.0)
            pitch = box_info.get("pitch", 0.0)
            yaw = box_info.get("yaw", box_info.get("theta", 0.0))
            z = box_info.get("z", 0.005 + h / 2)
            self.create_customized_box(
                name=name,
                color=box_info["color"],
                w=w,
                l=l,
                h=h,
                x=box_info["x"],
                y=box_info["y"],
                z=z,
                theta=roll,
                yaw=yaw,
                pitch=pitch,
            )
            self._graspable[name] = box_info.get(
                "graspable", max(w, l) <= self._max_grasp_width
            )
            body = self._objects[name]
            is_fixture = box_info.get("fixture", False) or box_info.get("static", False)
            if is_fixture:
                self._fixture_names.add(name)
            dynamics_config = {
                **box_info,
                "mass": 0.0 if is_fixture else box_info.get("mass", 1.0),
                "lateral_friction": box_info.get("lateral_friction", 0.7),
                "spinning_friction": box_info.get("spinning_friction", 0.02),
                "rolling_friction": box_info.get("rolling_friction", 0.01),
                "linear_damping": box_info.get(
                    "linear_damping",
                    0.85 if not self._graspable[name] else 0.20,
                ),
                "angular_damping": box_info.get(
                    "angular_damping",
                    0.90 if not self._graspable[name] else 0.20,
                ),
            }
            apply_configured_dynamics(
                body,
                dynamics_config,
                pybullet_module=p,
            )

        self.simulate()
        self._prepare_initial_joint_positions()
        return self.get_observation()

    def _create_push_backdrop(self):
        """Alias for ``create_studio_backdrop``."""
        return self.create_studio_backdrop()

    def _prepare_initial_joint_positions(self):
        """Normalize a scenario pre-push goal without teleporting the robot."""
        if self._initial_joint_positions is None:
            return

        requested = [float(value) for value in self._initial_joint_positions]
        joint_indices = list(self.robot.ik_joints)
        current = list(get_joint_positions(self.robot.robot, joint_indices))
        if len(requested) == 7 and len(joint_indices) >= 7:
            configured = requested + current[7:]
        elif len(requested) == len(joint_indices):
            configured = requested
        else:
            raise ValueError(
                "initial_joint_positions must contain either seven FR3 arm "
                f"joints or all {len(joint_indices)} controllable joints; "
                f"received {len(requested)}."
            )
        self._initial_joint_positions = tuple(configured)

    def is_fixture(self, name):
        return name in self._fixture_names or name.startswith("target_zone")

    def _normalize_push_targets(self):
        """Return {object_name: target_pose} from per-object or single-target config."""
        targets = {}
        object_targets = self._target_pose.get("object_targets")
        if object_targets:
            for name, pose in object_targets.items():
                merged = dict(self._target_pose)
                merged.pop("object_targets", None)
                merged.update(pose)
                targets[name] = merged
        elif self._stacking_order and {"x", "y"}.issubset(self._target_pose):
            targets[self._stacking_order[0]] = dict(self._target_pose)
        return targets

    def get_push_targets(self):
        return dict(self._object_targets)

    def get_push_target(self, obj_name):
        return self._object_targets.get(obj_name, self._target_pose)

    def get_push_bounds(self):
        bounds = self._target_pose.get("push_bounds", {})
        return {
            "x_min": bounds.get("x_min", 0.12),
            "x_max": bounds.get("x_max", 0.70),
            "y_min": bounds.get("y_min", -0.56),
            "y_max": bounds.get("y_max", 0.56),
            "z_floor": bounds.get("z_floor", 0.005),
        }

    def _is_pushable(self, obj_name):
        return obj_name in self.objects and not self.is_fixture(obj_name) and not self._graspable.get(obj_name, True)

    def _push_params(self, obj_name):
        params = dict(self._target_pose)
        params.update(self.get_push_target(obj_name))
        return params

    def _push_sequence_strings(self, obj_name, target_pose):
        actions = []
        theta = target_pose.get("theta", 0.0)
        for wp in target_pose.get("recommended_waypoints", []):
            actions.append(
                f"push(['{obj_name}'], {{'x': {wp[0]:.3f}, 'y': {wp[1]:.3f}, "
                f"'theta': {theta:.3f}}})"
            )
        actions.append(
            f"push(['{obj_name}'], {{'x': {target_pose['x']:.3f}, "
            f"'y': {target_pose['y']:.3f}, 'theta': {theta:.3f}}})"
        )
        return actions

    @staticmethod
    def _push_action_string(obj_name, x, y, theta=0.0):
        return (
            f"push(['{obj_name}'], {{'x': {x:.3f}, 'y': {y:.3f}, "
            f"'theta': {theta:.3f}}})"
        )

    def _multi_push_exchange_plan(self):
        """Compute a staged lane exchange for two mutually blocking push objects.

        This is derived from the live object footprints, starting poses, target
        poses, and configured push bounds.  It is intentionally not tied to a
        scenario name or fixed PUSH3 coordinates.
        """
        targets = self.get_push_targets()
        if self._target_pose.get("task_mode") != "multi_push" or len(targets) != 2:
            return []

        names = list(targets)
        if any(name not in self.objects for name in names):
            return []
        first_name, second_name = names
        first_start = np.asarray(self.get_position(first_name)[:2], dtype=float)
        second_start = np.asarray(self.get_position(second_name)[:2], dtype=float)
        first_target = np.asarray(
            [targets[first_name]["x"], targets[first_name]["y"]], dtype=float)
        second_target = np.asarray(
            [targets[second_name]["x"], targets[second_name]["y"]], dtype=float)

        # Only synthesize an exchange when each goal occupies the other
        # object's starting lane. Ordinary independent multi-push tasks retain
        # their direct symbolic plan.
        crossed = (
            abs(first_target[1] - second_start[1])
            < abs(first_target[1] - first_start[1])
            and abs(second_target[1] - first_start[1])
            < abs(second_target[1] - second_start[1])
        )
        if not crossed:
            return []

        first_min, first_max = self.get_bb(first_name)
        first_half_x = (first_max[0] - first_min[0]) / 2.0
        first_half_y = (first_max[1] - first_min[1]) / 2.0
        bounds = self.get_push_bounds()
        clearance = float(self._push_params(first_name).get("push_path_clearance", 0.010))

        # Pull the first object into a robot-side aisle, send the second object
        # directly to its now-clear target, then route the first object through
        # the opposite aisle to its own target.
        stage_x = max(
            bounds["x_min"] + first_half_x + clearance,
            min(first_start[0], second_start[0])
            - max(0.05, first_half_x * 0.75),
        )
        lower_lane = bounds["y_min"] + first_half_y + clearance
        upper_lane = bounds["y_max"] - first_half_y - clearance
        first_lane = lower_lane if first_start[1] <= second_start[1] else upper_lane
        opposite_lane = upper_lane if first_lane == lower_lane else lower_lane

        first_theta = float(targets[first_name].get("theta", 0.0))
        second_theta = float(targets[second_name].get("theta", 0.0))
        return [
            self._push_action_string(first_name, stage_x, first_lane, first_theta),
            self._push_action_string(
                second_name, second_target[0], second_target[1], second_theta),
            self._push_action_string(first_name, stage_x, opposite_lane, first_theta),
            self._push_action_string(
                first_name, first_target[0], first_target[1], first_theta),
        ]

    def get_push_detour_instruction(self, obj_name=None):
        targets = self.get_push_targets()
        exchange_plan = self._multi_push_exchange_plan()
        if exchange_plan and (obj_name is None or obj_name in targets):
            return "\n".join([
                "MANDATORY INTERLEAVED MULTI-PUSH EXCHANGE:",
                *[f"  {action}" for action in exchange_plan],
                "Execute these actions in this order so each colored box reaches its matching zone.",
            ])
        names = [obj_name] if obj_name in targets else list(targets.keys())
        lines = []
        for name in names:
            tp = targets.get(name)
            if not tp or not tp.get("recommended_waypoints"):
                continue
            sequence = "; ".join(self._push_sequence_strings(name, tp))
            lines.extend([
                f"MANDATORY DETOUR SEQUENCE for {name}: {sequence}.",
                (
                    "The next Full Plan must include these push actions in this exact order "
                    "before any pick/place. Do not retry the direct final-target push as the "
                    "first action after a straight-path-blocked rehearsal failure."
                ),
            ])
        return "\n".join(lines)

    def apply_action(self, action: Action, play_traj: bool = False):
        if action is None:
            return False, "No action is given!"
        if action.primitive not in self._primitive_actions.values():
            return False, "Unknown primitive action!"
        for obj_name in action.obj_args:
            if obj_name not in self.objects:
                return False, "Unknown object name!"

        traj = action.traj if action.traj is not None and len(action.traj) > 0 else None
        obj_name = action.obj_args[0]

        if action.primitive.name == "push":
            if self.is_fixture(obj_name):
                return False, f"{obj_name} is a static fixture/obstacle and cannot be pushed."
            success, traj, mp_feedback = self._execute_push(
                obj_name=obj_name,
                x=action.param_args["x"],
                y=action.param_args["y"],
                theta=action.param_args.get("theta", 0.0),
                traj=traj,
                play_traj=play_traj,
            )
        elif action.primitive.name == "pick":
            pending_pushes = [
                target_name
                for target_name in self.get_push_targets()
                if not self._base_in_target(target_name)
            ]
            push_before_pick = (
                self._target_pose.get("task_mode") != "multi_push"
                and bool(pending_pushes)
            )
            if push_before_pick:
                success, mp_feedback = False, (
                    "Push-before-pick precondition is not satisfied: "
                    f"{', '.join(pending_pushes)} must reach its push target "
                    f"before picking {obj_name}."
                )
            elif self.is_fixture(obj_name):
                success, mp_feedback = False, f"{obj_name} is a static fixture/obstacle and cannot be picked."
            elif not self._graspable.get(obj_name, True):
                bb_min, bb_max = self.get_bb(obj_name)
                width = max(bb_max[0] - bb_min[0], bb_max[1] - bb_min[1])
                success, mp_feedback = False, (
                    f"{obj_name} is not graspable: width {width:.3f}m exceeds "
                    f"gripper limit {self._max_grasp_width:.3f}m. Use push."
                )
            else:
                obj = self.objects[obj_name]
                obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)
                object_cfg = next(
                    (
                        cfg for cfg in self._objects_cfg.values()
                        if cfg["name"] == obj_name
                    ),
                    {},
                )
                success, traj, mp_feedback = self.robot.pick(
                    obj,
                    obstacles,
                    grasp_direction="top",
                    traj=traj,
                    play_traj=play_traj,
                    grasp_z_offset=object_cfg.get("grasp_z_offset"),
                    pregrasp_clearance=object_cfg.get(
                        "pregrasp_clearance", 0.08),
                )
        elif action.primitive.name == "place":
            obj = self.objects[obj_name]
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)
            z = action.param_args.get("z")
            if z is None:
                z = self._default_stack_place_z(obj_name)
            success, traj, mp_feedback = self.robot.place(
                obj,
                obstacles,
                x=action.param_args["x"],
                y=action.param_args["y"],
                z=z,
                theta=action.param_args.get("theta", 0.0),
                traj=traj,
                play_traj=play_traj,
            )
            if success:
                self.simulate()
                self.theta_dict[obj_name] = action.param_args.get("theta", 0.0)
        else:
            success, mp_feedback = False, "Unknown primitive action!"

        if action.traj is None or len(action.traj) == 0:
            action.traj = traj

        return success, mp_feedback

    def _check_push_path_clear(self, obj_name, start_xy, target_xy):
        """Reject straight pushes whose swept object footprint intersects obstacles."""
        bounds = self.get_push_bounds()
        clearance = float(self._push_params(obj_name).get("push_path_clearance", 0.010))
        bb_min, bb_max = self.get_bb(obj_name)
        half_x = (bb_max[0] - bb_min[0]) / 2.0
        half_y = (bb_max[1] - bb_min[1]) / 2.0

        obstacle_aabbs = []
        for other_name, other_id in self.objects.items():
            if other_name == obj_name or other_name == "table" or other_name.startswith("target_zone"):
                continue
            other_min, other_max = p.getAABB(other_id)
            obstacle_aabbs.append((other_name, other_min, other_max))

        distance = float(np.linalg.norm(target_xy - start_xy))
        samples = max(8, int(distance / 0.025))
        for t in np.linspace(0.0, 1.0, samples):
            center = (1.0 - t) * start_xy + t * target_xy
            bound_x_min = center[0] - half_x
            bound_x_max = center[0] + half_x
            bound_y_min = center[1] - half_y
            bound_y_max = center[1] + half_y
            x_min = bound_x_min - clearance
            x_max = bound_x_max + clearance
            y_min = bound_y_min - clearance
            y_max = bound_y_max + clearance

            eps = 1e-6
            if (bound_x_min < bounds["x_min"] - eps or bound_x_max > bounds["x_max"] + eps or
                    bound_y_min < bounds["y_min"] - eps or bound_y_max > bounds["y_max"] + eps):
                return False, (
                    f"Push path for {obj_name} leaves push bounds near "
                    f"({center[0]:.3f}, {center[1]:.3f})."
                )

            for other_name, other_min, other_max in obstacle_aabbs:
                overlaps = (
                    x_min < other_max[0] and x_max > other_min[0] and
                    y_min < other_max[1] and y_max > other_min[1]
                )
                if overlaps:
                    return False, (
                        f"Straight push path for {obj_name} is blocked by {other_name} "
                        f"near ({center[0]:.3f}, {center[1]:.3f}); use an ACFM-safe "
                        "intermediate push waypoint."
                    )
        return True, "Push path clear."

    @staticmethod
    def _validated_cached_push_segments(traj):
        """Validate the full transition/contact/retreat trace for a push."""
        raw_trace = getattr(traj, "motion_trace", None)
        if not raw_trace:
            return None
        segments = []
        for item in raw_trace:
            if not isinstance(item, dict) or not item.get("trajectory"):
                raise ValueError("Cached push motion trace contains a malformed segment")
            segments.append((item.get("phase"), list(item["trajectory"])))
        phases = [phase for phase, _ in segments]
        if phases.count("push_contact") != 1:
            raise ValueError("Cached push trace must contain one push_contact phase")
        contact_index = phases.index("push_contact")
        if contact_index == 0 and not segments[0][1]:
            raise ValueError("Cached push contact path is empty")
        allowed_pre = {
            "home_to_prepush",
            "prepush_to_contact",
            "current_to_push_contact",
        }
        if any(phase not in allowed_pre for phase in phases[:contact_index]):
            raise ValueError(f"Cached push trace has unsupported phases: {phases}")
        if phases[contact_index + 1:] != ["post_push_retreat"]:
            raise ValueError(
                "Cached push trace must end with one post_push_retreat phase"
            )
        traced = [conf for _, segment in segments for conf in segment]
        if len(traced) != len(traj) or not np.allclose(
            np.asarray(traced, dtype=float),
            np.asarray(traj, dtype=float),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "Cached push waypoints do not match the validated phase trace"
            )
        return segments

    def _execute_push(self, obj_name, x, y, theta, traj=None, play_traj=False):
        if self._graspable.get(obj_name, True):
            return False, traj, f"{obj_name} is graspable; use pick/place unless pushing is required."

        obj = self.objects[obj_name]
        target_xy = np.array([x, y], dtype=float)
        robot_body = self.robot.robot
        push_params = self._push_params(obj_name)
        push_tolerance = push_params.get("push_tolerance", 0.035)
        max_force = float(push_params.get("push_force_max", 16.0))
        min_force = float(push_params.get("push_force_min", 0.4))
        force_kp = float(push_params.get("push_force_kp", 120.0))
        force_kd = float(push_params.get("push_force_kd", 22.0))
        contact_count = 0
        executed_traj = []
        self.robot.last_motion_trace = []
        try:
            cached_segments = (
                self._validated_cached_push_segments(traj)
                if traj is not None else None
            )
        except ValueError as exc:
            return False, None, str(exc)
        def execute_planned_segment(phase, path):
            """Execute and record a planned transition without a state jump."""
            segment = [tuple(conf) for conf in path]
            if not segment:
                return
            if executed_traj and np.allclose(
                    executed_traj[-1], segment[0], atol=1e-8):
                segment = segment[1:]
            if not segment:
                return
            self.robot.simulate_traj(
                robot_body,
                self.robot.ik_joints,
                self.robot.attachments_robot,
                None,
                self.robot.tool_attach_link,
                None,
                segment,
                play_traj,
            )
            executed_traj.extend(segment)
            self.robot.last_motion_trace.append({
                "phase": phase,
                "trajectory": segment,
            })

        def apply_contact_force(contacts, direction, perp, contact_z):
            obj_pos, _ = p.getBasePositionAndOrientation(obj)
            obj_xy = np.array(obj_pos[:2])
            linear_vel, _ = p.getBaseVelocity(obj)
            planar_vel = np.array(linear_vel[:2])
            remaining = target_xy - obj_xy
            along_remaining = float(np.dot(remaining, direction))
            lateral_error = float(np.dot(remaining, perp))
            along_velocity = float(np.dot(planar_vel, direction))
            target_error = float(np.linalg.norm(remaining))

            if along_remaining > 0.004:
                raw_force_along = force_kp * along_remaining - force_kd * along_velocity
                force_along = float(np.clip(raw_force_along, 0.0, max_force))
                if 0.0 < force_along < min_force:
                    force_along = min_force
                if force_along > 0.0:
                    force_lateral = float(np.clip(35.0 * lateral_error, -5.0, 5.0))
                    force = direction * force_along + perp * force_lateral
                    force_point = [float(obj_pos[0]), float(obj_pos[1]), contact_z]
                    p.applyExternalForce(
                        obj,
                        -1,
                        [float(force[0]), float(force[1]), 0.0],
                        force_point,
                        p.WORLD_FRAME,
                    )

            return target_error, along_remaining, along_velocity

        start_margin = float(push_params.get("push_start_margin", 0.020))
        end_margin = float(push_params.get("push_end_margin", 0.018))
        overtravel = float(push_params.get("push_overtravel", 0.080))
        resolution = float(push_params.get("push_contact_resolution", 0.00025))
        lateral_gain = float(push_params.get("push_contact_lateral_gain", 0.5))
        max_lateral_adjust = float(push_params.get("push_max_lateral_adjust", 0.04))
        lost_contact_grace = int(push_params.get("push_lost_contact_grace", 6))
        ee_z_above_contact = float(push_params.get("push_ee_z_above_contact", 0.035))

        start_pos, _ = p.getBasePositionAndOrientation(obj)
        start_xy = np.array(start_pos[:2], dtype=float)
        delta = target_xy - start_xy
        distance = float(np.linalg.norm(delta))
        if distance <= push_tolerance:
            # Success with no motion: the object already satisfies this push,
            # so no contact phase is produced. Callers must treat this as a
            # satisfied no-op rather than a contactless failure.
            return True, traj or [], PUSH_ALREADY_AT_TARGET

        path_ok, path_feedback = self._check_push_path_clear(obj_name, start_xy, target_xy)
        if not path_ok:
            return False, traj, path_feedback

        direction = delta / distance
        perp = np.array([-direction[1], direction[0]])
        yaw = math.atan2(direction[1], direction[0])
        bb_min, bb_max = self.get_bb(obj_name)
        half_x = (bb_max[0] - bb_min[0]) / 2.0
        half_y = (bb_max[1] - bb_min[1]) / 2.0
        obj_h = bb_max[2] - bb_min[2]
        half_along = abs(direction[0]) * half_x + abs(direction[1]) * half_y
        contact_z = bb_min[2] + max(0.010, min(obj_h * 0.55, obj_h - 0.006))
        ee_z = min(bb_min[2] + obj_h - 0.004, contact_z + ee_z_above_contact)
        ee_orientation_candidates = [
            p.getQuaternionFromEuler([math.pi, 0.0, yaw]),
            p.getQuaternionFromEuler([math.pi, -0.25, yaw + math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, 0.0, yaw + math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, 0.25, yaw + math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, -0.25, yaw - math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, 0.0, yaw - math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, 0.25, yaw - math.pi / 2.0]),
            p.getQuaternionFromEuler([math.pi, -0.25, yaw]),
            p.getQuaternionFromEuler([math.pi, 0.25, yaw]),
            p.getQuaternionFromEuler([math.pi, 0.0, yaw + math.pi]),
        ]

        start_tool_xy = start_xy - direction * (half_along + start_margin)
        end_tool_xy = target_xy + direction * overtravel - direction * (half_along + end_margin)
        waypoint_count = max(120, int(distance / max(resolution, 1e-4)))
        lost_contact_steps = 0
        last_ik_failure = None
        collision_obstacles = self.prepare_obstacles(
            obj_name_list=[], remove_mode=True)
        escape_obstacles = self.prepare_obstacles(
            obj_name_list=[obj_name], remove_mode=True)
        self_collision_fn = get_collision_fn(
            robot_body,
            self.robot.ik_joints,
            obstacles=[],
            attachments=[],
            self_collisions=True,
        )
        approach_collision_fn = get_collision_fn(
            robot_body,
            self.robot.ik_joints,
            obstacles=list(collision_obstacles.keys()),
            attachments=[],
            self_collisions=True,
        )
        selected_push_orientation = None

        def solve_push_ik(ee_pos, require_world_clearance=False):
            nonlocal selected_push_orientation
            current_conf = get_joint_positions(robot_body, self.robot.ik_joints)
            last_reason = "no IK solution"
            orientations = (
                [selected_push_orientation]
                if selected_push_orientation is not None
                else ee_orientation_candidates
            )
            for ee_orn in orientations:
                increment_metric("ik_calls")
                conf = p.calculateInverseKinematics(
                    robot_body,
                    self.robot.tool_attach_link,
                    ee_pos,
                    ee_orn,
                    currentPositions=list(current_conf),
                    maxNumIterations=100000,
                    residualThreshold=0.0001,
                )
                if not self.robot.verify_ik(ee_pos, ee_orn, conf):
                    last_reason = "IK did not reach the requested push waypoint"
                    continue

                conf = tuple(conf)
                increment_metric("collision_checks")
                if self_collision_fn(conf):
                    set_joint_positions(robot_body, self.robot.ik_joints, current_conf)
                    last_reason = "IK posture is in robot self-collision"
                    continue

                if require_world_clearance:
                    increment_metric("collision_checks")
                if require_world_clearance and approach_collision_fn(conf):
                    set_joint_positions(robot_body, self.robot.ik_joints, current_conf)
                    last_reason = (
                        "IK posture collides with the push object or another "
                        "workspace obstacle"
                    )
                    continue

                set_joint_positions(robot_body, self.robot.ik_joints, current_conf)
                if selected_push_orientation is None:
                    selected_push_orientation = ee_orn
                return conf, None

            set_joint_positions(robot_body, self.robot.ik_joints, current_conf)
            return None, last_reason

        if cached_segments is not None:
            replay_start = cached_segments[0][1][0]
            current_conf = tuple(get_joint_positions(
                robot_body, self.robot.ik_joints))
            if not np.allclose(
                np.asarray(replay_start, dtype=float),
                np.asarray(current_conf, dtype=float),
                atol=1e-6,
                rtol=0.0,
            ):
                return False, None, (
                    "Cached push trajectory start-state mismatch; discard the "
                    "cached path and invoke motion planning from the current state"
                )

        # A configured measured posture is a pre-push goal, not an initial
        # state override. Plan to it from the actual reset/home state.
        # The pushed object remains a collision obstacle during every global
        # transition. It is exempted only for the short retreat that begins in
        # intentional contact, and that retreat is penetration-audited below.
        use_configured_prepush = (
            self._initial_joint_positions is not None
            and not self._prepush_transition_done
        )

        def execute_contact_conf(conf):
            nonlocal contact_count, lost_contact_steps
            executed_traj.append(conf)
            set_joint_positions(robot_body, self.robot.ik_joints, conf)
            contacts = p.getContactPoints(bodyA=robot_body, bodyB=obj)
            if contacts:
                lost_contact_steps = 0
                contact_count += len(contacts)
                apply_contact_force(contacts, direction, perp, contact_z)
            else:
                lost_contact_steps += 1
                if contact_count == 0 or lost_contact_steps > lost_contact_grace:
                    p.resetBaseVelocity(
                        obj,
                        linearVelocity=[0.0, 0.0, 0.0],
                        angularVelocity=[0.0, 0.0, 0.0],
                    )
            increment_metric("simulator_steps")
            p.stepSimulation()
            if play_traj:
                replay_capture.frame()
                wait_for_duration(self.robot.trajectory_time_step)

        if cached_segments is not None:
            contact_index = next(
                index for index, (phase, _) in enumerate(cached_segments)
                if phase == "push_contact"
            )
            for phase, segment in cached_segments[:contact_index]:
                execute_planned_segment(phase, segment)
                if phase == "home_to_prepush":
                    self._prepush_transition_done = True
            contact_path = cached_segments[contact_index][1]
        else:
            if use_configured_prepush:
                prepush_success, prepush_traj, prepush_feedback = (
                    self.robot.plan_joint_configuration(
                        self._initial_joint_positions,
                        collision_obstacles,
                        self.robot.attachments_robot,
                    )
                )
                if not prepush_success:
                    return False, traj, (
                        "Motion planning from home to the configured pre-push "
                        f"posture failed: {prepush_feedback}"
                    )
                execute_planned_segment("home_to_prepush", prepush_traj)
                self._prepush_transition_done = True

            # The closed-loop waypoint/IK generator is itself one planner
            # invocation; nested global transitions are counted separately by
            # PyBulletRobot.plan_joint_configuration/motion_planning.  Count
            # it only when generation actually starts, after hard path checks
            # and any required pre-push transition have succeeded.
            self.robot._record_motion_planner_call()
            for i in range(waypoint_count):
                t = i / max(1, waypoint_count - 1)
                nominal_xy = (1.0 - t) * start_tool_xy + t * end_tool_xy
                obj_xy = np.array(p.getBasePositionAndOrientation(obj)[0][:2])
                lateral_error = float(np.dot(target_xy - obj_xy, perp))
                lateral_adjust = float(
                    np.clip(lateral_gain * lateral_error, -max_lateral_adjust, max_lateral_adjust)
                )
                xy = nominal_xy + perp * lateral_adjust
                ee_pos = [float(xy[0]), float(xy[1]), ee_z]
                if i == 0:
                    # First-contact IK validation temporarily assigns candidate
                    # configurations. Keep those planning-only poses invisible;
                    # only the subsequent collision-free transition is played.
                    with LockRenderer():
                        conf, ik_failure = solve_push_ik(
                            ee_pos, require_world_clearance=True)
                else:
                    conf, ik_failure = solve_push_ik(
                        ee_pos, require_world_clearance=False)
                if conf is None:
                    msg = (
                        f"No collision-free IK solution for push waypoint at "
                        f"({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}); "
                        f"{ik_failure}."
                    )
                    if len(executed_traj) == 0:
                        return False, traj, msg
                    last_ik_failure = msg
                    break

                if i == 0:
                    current_conf = tuple(get_joint_positions(
                        robot_body, self.robot.ik_joints))
                    if not np.allclose(current_conf, conf, atol=1e-8):
                        contact_success, contact_traj, contact_feedback = (
                            self.robot.plan_joint_configuration(
                                conf,
                                collision_obstacles,
                                self.robot.attachments_robot,
                            )
                        )
                        if not contact_success:
                            return False, executed_traj or traj, (
                                "Motion planning from the pre-push posture to "
                                f"first push contact failed: {contact_feedback}"
                            )
                        phase = (
                            "prepush_to_contact"
                            if use_configured_prepush
                            else "current_to_push_contact"
                        )
                        execute_planned_segment(phase, contact_traj)
                execute_contact_conf(conf)
                current_xy = np.array(
                    p.getBasePositionAndOrientation(obj)[0][:2])
                if float(np.linalg.norm(current_xy - target_xy)) <= push_tolerance:
                    break

        if cached_segments is not None:
            for conf in contact_path:
                execute_contact_conf(conf)

        push_segment_start = 0
        for trace_segment in self.robot.last_motion_trace:
            push_segment_start += len(trace_segment["trajectory"])
        push_segment = [tuple(conf) for conf in executed_traj[push_segment_start:]]
        if push_segment:
            self.robot.last_motion_trace.append({
                "phase": "push_contact",
                "trajectory": push_segment,
            })
        traj = executed_traj
        p.resetBaseVelocity(
            obj,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )
        self.simulate(60)

        final_pos, final_orn = p.getBasePositionAndOrientation(obj)
        retreat_distance = float(push_params.get("push_retreat_distance", 0.080))
        retreat_lift = float(push_params.get("push_retreat_lift", 0.060))
        current_ee_pos, current_ee_orn = get_link_pose(
            robot_body, self.robot.tool_attach_link)
        retreat_target = np.array([
            current_ee_pos[0] - direction[0] * retreat_distance,
            current_ee_pos[1] - direction[1] * retreat_distance,
            min(current_ee_pos[2] + retreat_lift, 0.58),
        ])
        if cached_segments is not None:
            retreat_success = True
            retreat_traj = cached_segments[-1][1]
            retreat_feedback = "Success"
        else:
            retreat_success, retreat_traj, retreat_feedback = (
                self.robot.motion_planning(
                    robot_body,
                    self.robot.ik_joints,
                    retreat_target,
                    current_ee_orn,
                    escape_obstacles,
                    self.robot.attachments_robot,
                    self.robot.planning_limits,
                )
            )
        if not retreat_success:
            return False, traj, (
                "Push contact completed, but motion planning for the "
                f"post-push clearance retreat failed: {retreat_feedback}"
            )
        saved_retreat_start = tuple(get_joint_positions(
            robot_body, self.robot.ik_joints))
        worst_penetration = 0.0
        with LockRenderer():
            try:
                for retreat_conf in retreat_traj:
                    set_joint_positions(
                        robot_body, self.robot.ik_joints, retreat_conf)
                    increment_metric("collision_checks")
                    contacts = p.getClosestPoints(robot_body, obj, distance=0.0)
                    if contacts:
                        worst_penetration = min(
                            worst_penetration,
                            min(float(contact[8]) for contact in contacts),
                        )
            finally:
                set_joint_positions(
                    robot_body, self.robot.ik_joints, saved_retreat_start)
        max_allowed_penetration = float(
            push_params.get("push_max_retreat_penetration", 0.0005))
        if worst_penetration < -max_allowed_penetration:
            return False, traj, (
                "Rejected post-push retreat because it would penetrate "
                f"{obj_name} by {-worst_penetration:.4f}m."
            )
        execute_planned_segment("post_push_retreat", retreat_traj)
        if contact_count == 0:
            return False, traj, (
                f"Push failed because the gripper never contacted {obj_name}; "
                "adjust the push approach or object pose."
            )

        final_xy = np.array(final_pos[:2])
        error = float(np.linalg.norm(final_xy - target_xy))
        if error > push_params.get("push_tolerance", 0.035):
            extra = f" ({last_ik_failure})." if last_ik_failure else "."
            return False, traj, (
                f"Push ended {error:.3f}m from target "
                f"(target=({x:.3f}, {y:.3f}), actual=({final_xy[0]:.3f}, {final_xy[1]:.3f}))"
                f"{extra}"
            )

        if theta is not None:
            yaw_now = p.getEulerFromQuaternion(final_orn)[2]
            yaw_error = abs(math.atan2(math.sin(yaw_now - theta), math.cos(yaw_now - theta)))
            if yaw_error > push_params.get("theta_tolerance", 0.35):
                return False, traj, (
                    f"Push reached target but yaw error is {yaw_error:.3f}rad; "
                    f"target theta={theta:.3f}, actual={yaw_now:.3f}."
                )

        self.theta_dict[obj_name] = theta
        return True, traj, f"Success (gripper contacts={contact_count})"

    def _default_stack_place_z(self, obj_name):
        if len(self._stacking_order) < 2:
            return 0.05
        base_name = self._stacking_order[0]
        base_top = self.get_bb(base_name)[1][2]
        obj_min, obj_max = self.get_bb(obj_name)
        # Plan the attached object just above the support surface so collision
        # checking does not reject intentional final contact. After release,
        # normal simulation settles it onto the base.
        placement_clearance = float(
            self._target_pose.get("stack_place_clearance", 0.01))
        return (
            base_top
            + (obj_max[2] - obj_min[2]) / 2.0
            + placement_clearance
        )

    def get_observation(self):
        observation = super().get_observation()
        fixture_states = {
            key: value for key, value in observation.items()
            if self.is_fixture(key) and key not in {"table"} and not key.startswith("target_zone")
        }
        for key in list(observation.keys()):
            if self.is_fixture(key):
                observation.pop(key, None)

        targets = self.get_push_targets()
        first_target = next(iter(targets.values()), self._target_pose)
        target_text = (
            "Target push poses: " +
            "; ".join(
                f"{name} -> x={tp['x']:.3f}, y={tp['y']:.3f}, theta={tp.get('theta', 0.0):.3f}"
                for name, tp in targets.items()
            )
        )
        if not targets:
            target_text = (
                f"Target push/stack pose: x={first_target['x']:.3f}, "
                f"y={first_target['y']:.3f}, theta={first_target.get('theta', 0.0):.3f}."
            )
        if self._approach_tcp_pose is not None:
            target_text += (
                "\nConfigured initial push approach TCP position: "
                f"{textualize_array(np.asarray(self._approach_tcp_pose, dtype=float))}."
            )
        if self._initial_joint_positions is not None:
            target_text += (
                "\nConfigured FR3 pre-push arm joint goal (planned from home): "
                f"{textualize_array(np.asarray(self._initial_joint_positions, dtype=float))}."
            )

        object_text = f"Objects: {', '.join(observation.keys())}."
        for obj_name, obj_state in observation.items():
            bb_min = obj_state["bb_min"]
            bb_max = obj_state["bb_max"]
            pos = obj_state["position"]
            w = bb_max[0] - bb_min[0]
            l = bb_max[1] - bb_min[1]
            h = bb_max[2] - bb_min[2]
            affordance = "graspable" if self._graspable.get(obj_name, True) else "not graspable; must be pushed"
            object_text += (
                f"\n{obj_name} is at {textualize_array(pos)}, "
                f"size (x={w:.3f}, y={l:.3f}, z={h:.3f}), "
                f"bounding box [{textualize_array(bb_min)} to {textualize_array(bb_max)}], "
                f"affordance: {affordance}."
            )
        if fixture_states:
            object_text += "\nStatic obstacles/fixtures:"
            for name, state in fixture_states.items():
                bb_min = state["bb_min"]
                bb_max = state["bb_max"]
                pos = state["position"]
                object_text += (
                    f"\n{name} is fixed at {textualize_array(pos)}, "
                    f"bounding box [{textualize_array(bb_min)} to {textualize_array(bb_max)}]."
                )

        detour_text = self.get_push_detour_instruction()
        plan_text = self._recommended_plan_text()
        status_text = self._status_text()
        return observation, "\n".join(
            text for text in [target_text, object_text, detour_text, plan_text, status_text]
            if text
        )

    def _recommended_plan_text(self):
        targets = self.get_push_targets()
        if not targets:
            return ""

        exchange_plan = self._multi_push_exchange_plan()
        if exchange_plan:
            return (
                "Recommended collision-free multi-push exchange plan: "
                + "; ".join(exchange_plan) + "."
            )

        action_text = []
        for obj_name, tp in targets.items():
            waypoints = tp.get("recommended_waypoints", [])
            for wp in waypoints:
                action_text.append(
                    f"push(['{obj_name}'], {{'x': {wp[0]:.3f}, 'y': {wp[1]:.3f}, "
                    f"'theta': {tp.get('theta', 0.0):.3f}}})"
                )
            action_text.append(
                f"push(['{obj_name}'], {{'x': {tp['x']:.3f}, 'y': {tp['y']:.3f}, "
                f"'theta': {tp.get('theta', 0.0):.3f}}})"
            )

        if self._target_pose.get("task_mode") == "multi_push":
            return "Recommended multi-push plan skeleton: " + "; ".join(action_text) + "."

        if len(self._stacking_order) >= 2:
            base_name, top_name = self._stacking_order[:2]
            base_target = targets.get(base_name, next(iter(targets.values())))
            z = self._default_stack_place_z(top_name)
            action_text.extend([
                f"pick(['{top_name}'], {{}})",
                (
                    f"place(['{top_name}'], {{'x': {base_target['x']:.3f}, "
                    f"'y': {base_target['y']:.3f}, 'z': {z:.3f}, "
                    f"'theta': {base_target.get('theta', 0.0):.3f}}})"
                ),
            ])
        return "Recommended push-stack plan skeleton: " + "; ".join(action_text) + "."

    def _status_text(self):
        targets = self.get_push_targets()
        if not targets:
            return "Current status: incomplete task definition."
        target_status = [
            f"{name}: {'at target' if self._base_in_target(name) else 'not at target'}"
            for name in targets
        ]
        if self._target_pose.get("task_mode") == "multi_push" or len(self._stacking_order) < 2:
            return "Current status: " + ", ".join(target_status) + "."
        base_name, top_name = self._stacking_order[:2]
        top_on_base, detail = self._top_on_base(top_name, base_name)
        return (
            "Current status: " + ", ".join(target_status) + ", "
            f"{top_name}: {'on base' if top_on_base else detail}."
        )

    def get_foresight_guidance(self):
        targets = self.get_push_targets()
        if not targets:
            return ""
        lines = [
            "=== PUSH-STACK AFFORDANCE MAP ===",
        ]
        for obj_name, tp in targets.items():
            lines.append(
                f"{obj_name}: not graspable. pick(['{obj_name}'], {{}}) should be avoided; use push."
            )
            lines.append(
                f"Primary push target: push(['{obj_name}'], "
                f"{{'x': {tp['x']:.3f}, 'y': {tp['y']:.3f}, "
                f"'theta': {tp.get('theta', 0.0):.3f}}})."
            )
            waypoints = tp.get("recommended_waypoints", [])
            if waypoints:
                wp_text = ", ".join(f"({wp[0]:.3f}, {wp[1]:.3f})" for wp in waypoints)
                lines.append(
                    f"Known safe intermediate waypoint(s) for {obj_name}: {wp_text}. "
                    "Use them only when direct ACFM/rehearsal says the straight push is blocked."
                )
                lines.append(self.get_push_detour_instruction(obj_name))
        lines.extend([
            (
                "Push feasibility: the push ACFM scores each candidate endpoint by swept-path "
                "collision, final-footprint collision, gripper-contact reachability, progress "
                "toward the final target, table-bound stability, and failure history. "
                "The primitive uses actual gripper contact and applies force at "
                "contact points while continuously adjusting lateral contact."
            ),
        ])
        if self._target_pose.get("task_mode") != "multi_push" and len(self._stacking_order) >= 2:
            base_name, top_name = self._stacking_order[:2]
            tp = targets.get(base_name, next(iter(targets.values())))
            z = self._default_stack_place_z(top_name)
            lines.append(
                f"After the push, place {top_name} on top at "
                f"({tp['x']:.3f}, {tp['y']:.3f}, z={z:.3f}, "
                f"theta={tp.get('theta', 0.0):.3f})."
            )
        lines.append("=== END PUSH-STACK AFFORDANCE MAP ===")
        return "\n".join(lines)

    def _base_in_target(self, base_name):
        if base_name not in self.objects:
            return False
        pos = self.get_position(base_name)
        tp = self.get_push_target(base_name)
        dist = math.hypot(pos[0] - tp["x"], pos[1] - tp["y"])
        return dist <= tp.get("radius", 0.05)

    def _top_on_base(self, top_name, base_name):
        if top_name not in self.objects or base_name not in self.objects:
            return False, "missing object"
        top_min, top_max = self.get_bb(top_name)
        base_min, base_max = self.get_bb(base_name)
        base_top = base_max[2]
        z_gap = top_min[2] - base_top
        top_cx = (top_min[0] + top_max[0]) / 2.0
        top_cy = (top_min[1] + top_max[1]) / 2.0
        base_cx = (base_min[0] + base_max[0]) / 2.0
        base_cy = (base_min[1] + base_max[1]) / 2.0
        xy_dist = math.hypot(top_cx - base_cx, top_cy - base_cy)
        if abs(z_gap) > 0.03:
            return False, f"not on base (z gap={z_gap:.3f}m)"
        stack_xy_tolerance = float(self._target_pose.get("stack_xy_tolerance", 0.04))
        if xy_dist > stack_xy_tolerance:
            return False, f"not centered on base (offset={xy_dist:.3f}m)"
        return True, "on base"

    def check_goal(self):
        targets = self.get_push_targets()
        if not targets:
            return False, "Need at least one push target."
        issues = []
        for base_name in targets:
            if not self._base_in_target(base_name):
                pos = self.get_position(base_name)
                issues.append(
                    f"{base_name} not at target (actual=({pos[0]:.3f}, {pos[1]:.3f}))"
                )
        if self._target_pose.get("task_mode") != "multi_push" and len(self._stacking_order) >= 2:
            base_name, top_name = self._stacking_order[:2]
            top_ok, top_detail = self._top_on_base(top_name, base_name)
            if not top_ok:
                issues.append(f"{top_name} {top_detail}")
        if issues:
            return False, "; ".join(issues)
        if self._target_pose.get("task_mode") == "multi_push":
            return True, "All push objects reached their targets."
        return True, "Large base pushed to target and small object stacked on top."

    def describe_placement(self, x, y):
        """Name a push destination so a candidate can be matched to a plan step.

        Push targets are continuous, so the name is the waypoint itself, as
        already stated in the observation's detour sequence.
        """
        return f"({float(x):.3f}, {float(y):.3f})"

    def get_symbolic_plan(self):
        targets = self.get_push_targets()
        if not targets:
            return []
        exchange_plan = self._multi_push_exchange_plan()
        if exchange_plan:
            return exchange_plan
        plan = []
        for obj_name, tp in targets.items():
            for wp in tp.get("recommended_waypoints", []):
                plan.append(
                    f"push(['{obj_name}'], {{'x': {wp[0]:.3f}, 'y': {wp[1]:.3f}, 'theta': {tp.get('theta', 0.0):.3f}}})"
                )
            plan.append(
                f"push(['{obj_name}'], {{'x': {tp['x']:.3f}, 'y': {tp['y']:.3f}, 'theta': {tp.get('theta', 0.0):.3f}}})"
            )
        if self._target_pose.get("task_mode") != "multi_push" and len(self._stacking_order) >= 2:
            base_name, top_name = self._stacking_order[:2]
            tp = targets.get(base_name, next(iter(targets.values())))
            z = self._default_stack_place_z(top_name)
            plan.extend([
                f"pick(['{top_name}'], {{}})",
                f"place(['{top_name}'], {{'x': {tp['x']:.3f}, 'y': {tp['y']:.3f}, 'z': {z:.3f}, 'theta': {tp.get('theta', 0.0):.3f}}})",
            ])
        return plan

    def create_task_instances(self, *args, **kwargs):
        pass
