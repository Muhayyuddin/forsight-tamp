import logging
import numpy as np

from envs.constants import COLOR_MAP, TAN
from envs.pb_env import PybulletEnv
from utils.io_util import load_json, dump_json
from utils.tamp_util import Action, PrimitiveAction, Parameter
from utils.llm_util import textualize_array
from utils.dynamics_util import apply_configured_dynamics
from utils.pb_util import create_box, set_point

import pybullet as p

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StackingEnv(PybulletEnv):
    """
    Environment for stacking tasks (Tower of Hanoi, block towers, pyramids).
    Objects sit on a table surface. No basket — the goal is stacking order, not containment.
    A target zone on the table marks where the stack should be built.
    """

    def __init__(self):
        super().__init__()

        self._primitive_actions = {
            "pick": PrimitiveAction(name="pick", obj_arity=1),
            "place": PrimitiveAction(
                name="place",
                obj_arity=1,
                parameters={
                    "x": Parameter("x", lower_limit=0.0, upper_limit=1.4),
                    "y": Parameter("y", lower_limit=-1.0, upper_limit=1.0),
                    "z": Parameter("z", lower_limit=0.0, upper_limit=0.5),
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
        }

    def __str__(self):
        return "StackingEnv"

    def reset(self, boxes, stacking_order, use_gui=True,
              stacking_zone=None, **kwargs):
        """Reset environment for a stacking task.

        Args:
            boxes: dict of box configs (name, color, w, l, h, x, y, z, ...)
            stacking_order: list of object names bottom-to-top
            stacking_zone: optional dict {x, y, radius} marking target area
        """
        super().reset(use_gui=use_gui)

        # Store task config
        self._stacking_order = stacking_order
        self._stacking_zone = stacking_zone or {"x": 0.55, "y": 0.0, "radius": 0.15}
        self._box_specs = {
            box_info["name"]: dict(box_info)
            for box_info in boxes.values()
            if "name" in box_info
        }

        # Create table
        self.create_table()

        # Draw target zone marker on the table (thin colored circle-like square)
        sz = self._stacking_zone
        marker = create_box(
            w=sz["radius"] * 2, l=sz["radius"] * 2, h=0.002,
            color=[0.4, 0.7, 0.4, 0.4],
        )
        set_point(marker, [sz["x"], sz["y"], 0.006])
        p.changeDynamics(marker, -1, mass=0)
        # This translucent plate is a visual annotation, not geometry in the
        # manipulation problem.  Disable contacts and also omit it from motion
        # obstacles below so collision queries cannot reject the intended
        # stack location.
        p.setCollisionFilterGroupMask(marker, -1, 0, 0)
        self._objects["target_zone"] = marker

        # Create objects
        for box_info in boxes.values():
            yaw = box_info.get("yaw", 0.0)
            ycb_urdf = box_info.get("ycb_urdf")
            if ycb_urdf:
                scaling = box_info.get("global_scaling", 0.07)
                loaded = self.create_ycb_object(
                    name=box_info["name"],
                    ycb_urdf=ycb_urdf,
                    x=box_info["x"],
                    y=box_info["y"],
                    z=box_info["z"],
                    theta=np.pi,
                    global_scaling=scaling,
                    yaw=yaw,
                )
                if not loaded:
                    self.create_customized_box(
                        name=box_info["name"],
                        color=box_info["color"],
                        w=box_info["w"],
                        l=box_info["l"],
                        h=box_info["h"],
                        x=box_info["x"],
                        y=box_info["y"],
                        z=box_info["z"],
                        theta=np.pi,
                        yaw=yaw,
                    )
            else:
                self.create_customized_box(
                    name=box_info["name"],
                    color=box_info["color"],
                    w=box_info["w"],
                    l=box_info["l"],
                    h=box_info["h"],
                    x=box_info["x"],
                    y=box_info["y"],
                    z=box_info["z"],
                    theta=np.pi,
                    yaw=yaw,
                )
            apply_configured_dynamics(
                self._objects[box_info["name"]],
                box_info,
                pybullet_module=p,
            )

        # Settle physics
        self.simulate()

        observation = self.get_observation()
        return observation

    def apply_action(self, action: Action, play_traj: bool = False):
        if action is None:
            return False, "No action is given!"
        if action.primitive not in self._primitive_actions.values():
            return False, "Unknown primitive action!"
        for obj_name in action.obj_args:
            if obj_name not in self.objects:
                return False, "Unknown object name!"

        traj = action.traj if action.traj is not None and len(action.traj) > 0 else None

        if action.primitive.name == "pick":
            obj_name = action.obj_args[0]
            obj = self.objects[obj_name]
            obstacles = self._prepare_motion_obstacles(obj_name)
            object_cfg = self._box_specs.get(obj_name, {})

            success, traj, mp_feedback = self.robot.pick(
                obj,
                obstacles,
                grasp_direction="top",
                traj=traj,
                play_traj=play_traj,
                grasp_yaw=object_cfg.get("grasp_theta", 0.0),
                grasp_z_offset=object_cfg.get("grasp_z_offset"),
                # A top grasp made directly at table height leaves the newly
                # attached object in intentional table contact, which makes
                # the following transfer planner start in collision.  Use the
                # same vertical approach/departure contract as packing so the
                # attachment clears its support before any lateral motion.
                pregrasp_clearance=object_cfg.get(
                    "pregrasp_clearance", 0.08),
            )
            if success:
                logger.debug("Picked!")
            else:
                logger.debug(f"Pick failed: {mp_feedback}")

        elif action.primitive.name == "place":
            obj_name = action.obj_args[0]
            obj = self.objects[obj_name]
            obstacles = self._prepare_motion_obstacles(obj_name)
            requested_pose = {
                "x": action.param_args["x"],
                "y": action.param_args["y"],
                "z": action.param_args.get("z", 0.05),
            }
            target_pose, pose_feedback = self._authoritative_stack_place_pose(
                obj_name, requested_pose)
            if target_pose is None:
                return False, pose_feedback

            # Keep the action object in sync with the enforced physical target
            # so logs, rehearsal, and later analysis describe the real placement.
            action.param_args["x"] = target_pose["x"]
            action.param_args["y"] = target_pose["y"]
            action.param_args["z"] = target_pose["z"]
            action.param_args["theta"] = action.param_args.get("theta", 0.0)

            support_name = target_pose.get("support_name")
            if support_name != "table" and support_name in self.objects:
                # The support object is intentionally touched by the placed
                # object; treat it as support, not as a blocking obstacle.
                obstacles.pop(self.objects[support_name], None)

            success, traj, mp_feedback = self.robot.place(
                obj,
                obstacles,
                x=target_pose["x"],
                y=target_pose["y"],
                z=target_pose["z"],
                theta=action.param_args["theta"],
                traj=traj,
                play_traj=play_traj,
            )
            if success:
                self.simulate()
                self.theta_dict[obj_name] = action.param_args["theta"]
                logger.debug("Placed!")
            else:
                logger.debug(f"Place failed: {mp_feedback}")

        if action.traj is None or len(action.traj) == 0:
            action.traj = traj

        return success, mp_feedback

    def _prepare_motion_obstacles(self, manipulated_name):
        """Return physical obstacles, excluding the visual target marker."""
        obstacles = self.prepare_obstacles(
            obj_name_list=[manipulated_name], remove_mode=True)
        marker = self.objects.get("target_zone")
        if marker is not None:
            obstacles.pop(marker, None)
        return obstacles

    def _body_z_with_bottom_on(self, obj_name, support_z, clearance=0.003):
        """Body-base z that puts the object's AABB bottom on a support surface."""
        obj = self.objects[obj_name]
        aabb = p.getAABB(obj)
        body_z = p.getBasePositionAndOrientation(obj)[0][2]
        aabb_center_z = (aabb[0][2] + aabb[1][2]) / 2.0
        aabb_offset_z = aabb_center_z - body_z
        half_z = (aabb[1][2] - aabb[0][2]) / 2.0
        return float(support_z + clearance + half_z - aabb_offset_z)

    def _authoritative_stack_place_pose(self, obj_name, requested_pose=None):
        """Return the exact stacking target for obj_name.

        For stack tasks, high-score ACFM cells describe feasible regions, but
        the final structure requires exact center alignment. This method makes
        the environment authoritative: base object goes to the target-zone
        center, every later object goes to the previous object's center/top.
        """
        requested_pose = requested_pose or {}
        if obj_name not in getattr(self, "_stacking_order", []):
            return {
                "x": float(requested_pose.get("x", 0.0)),
                "y": float(requested_pose.get("y", 0.0)),
                "z": float(requested_pose.get("z", 0.05)),
                "support_name": None,
            }, "Using planner-specified placement"

        stack_idx = self._stacking_order.index(obj_name)
        if stack_idx == 0:
            support_z = self.get_bb("table")[1][2] if "table" in self.objects else 0.005
            return {
                "x": float(self._stacking_zone["x"]),
                "y": float(self._stacking_zone["y"]),
                "z": self._body_z_with_bottom_on(obj_name, support_z),
                "support_name": "table",
            }, "Success"

        prev_name = self._stacking_order[stack_idx - 1]
        if prev_name not in self.objects:
            return None, f"Cannot place {obj_name}: support object {prev_name} is missing"
        if not self._is_in_zone(prev_name):
            return None, (
                f"Cannot place {obj_name} before {prev_name} is stacked in the target zone"
            )

        prev_bb_min, prev_bb_max = self.get_bb(prev_name)
        return {
            "x": float((prev_bb_min[0] + prev_bb_max[0]) / 2.0),
            "y": float((prev_bb_min[1] + prev_bb_max[1]) / 2.0),
            "z": self._body_z_with_bottom_on(obj_name, prev_bb_max[2]),
            "support_name": prev_name,
        }, "Success"

    def get_observation(self):
        observation = super().get_observation()

        # Remove non-task objects from observation
        observation.pop("table", None)
        observation.pop("target_zone", None)

        if not observation:
            return observation, ""

        # Stacking zone info
        sz = self._stacking_zone
        zone_text = (
            f"There is a target stacking zone centered at "
            f"({sz['x']:.2f}, {sz['y']:.2f}) with radius {sz['radius']:.2f}m. "
            f"Build the stack in this zone."
        )

        # Objects info
        names = list(observation.keys())
        obj_text = f"Objects to stack: {', '.join(names)}."
        for obj_name, obj_state in observation.items():
            pos = obj_state["position"]
            bb_min = obj_state["bb_min"]
            bb_max = obj_state["bb_max"]
            w = bb_max[0] - bb_min[0]
            l = bb_max[1] - bb_min[1]
            h = bb_max[2] - bb_min[2]
            obj_text += (
                f"\n{obj_name} is at position {textualize_array(pos)}, "
                f"bounding box [{textualize_array(bb_min)} to {textualize_array(bb_max)}], "
                f"size (w={w:.3f}, l={l:.3f}, h={h:.3f})."
            )

        # Stacking order
        order_text = (
            f"Required stacking order (bottom to top): "
            f"{' -> '.join(self._stacking_order)}."
        )

        # Current stacking status
        status_parts = []
        for i, obj_name in enumerate(self._stacking_order):
            if obj_name not in self.objects:
                status_parts.append(f"{obj_name}: missing")
                continue
            bb_min, bb_max = self.get_bb(obj_name)
            pos = self.get_position(obj_name)
            on_table = bb_min[2] < 0.02
            if on_table:
                in_zone = self._is_in_zone(obj_name)
                status_parts.append(
                    f"{obj_name}: on table" +
                    (" (in zone)" if in_zone else " (outside zone)")
                )
            elif i > 0:
                prev = self._stacking_order[i - 1]
                if prev in self.objects:
                    prev_top = self.get_bb(prev)[1][2]
                    if abs(bb_min[2] - prev_top) < 0.03:
                        status_parts.append(f"{obj_name}: on {prev}")
                    else:
                        status_parts.append(
                            f"{obj_name}: elevated (z={pos[2]:.3f})")
                else:
                    status_parts.append(f"{obj_name}: elevated (z={pos[2]:.3f})")
            else:
                status_parts.append(f"{obj_name}: elevated (z={pos[2]:.3f})")

        status_text = "Current status: " + ", ".join(status_parts) + "."

        obs_text = "\n".join([zone_text, obj_text, order_text, status_text])
        return observation, obs_text

    def _is_in_zone(self, obj_name, tol=0.02):
        """Check if object center is within the stacking zone."""
        pos = self.get_position(obj_name)
        sz = self._stacking_zone
        dist = np.sqrt((pos[0] - sz["x"])**2 + (pos[1] - sz["y"])**2)
        return dist < sz["radius"] + tol

    def check_stacking_order(self, stacking_order=None, tol=0.02):
        """Check if objects are stacked in the specified order (bottom to top)."""
        stacking_order = stacking_order or self._stacking_order
        issues = []

        for i, obj_name in enumerate(stacking_order):
            if obj_name not in self.objects or obj_name in ("table", "target_zone"):
                issues.append(f"{obj_name} not found")
                continue

            bb_min, bb_max = self.get_bb(obj_name)

            if i == 0:
                # Bottom object should be on the table surface
                if bb_min[2] > 0.03:
                    issues.append(
                        f"{obj_name} (base) not on table (z_min={bb_min[2]:.3f})")
                # Should be in the target zone
                if not self._is_in_zone(obj_name):
                    issues.append(f"{obj_name} (base) not in target zone")
            else:
                prev_name = stacking_order[i - 1]
                if prev_name not in self.objects:
                    continue
                prev_bb_min, prev_bb_max = self.get_bb(prev_name)

                # Check z-gap: object bottom should be near previous object top
                z_gap = bb_min[2] - prev_bb_max[2]
                if z_gap > tol:
                    issues.append(
                        f"{obj_name} not on {prev_name} (gap={z_gap:.3f}m)")
                elif z_gap < -tol:
                    issues.append(
                        f"{obj_name} intersects {prev_name}")

                # Check xy centering
                cx = (bb_min[0] + bb_max[0]) / 2
                cy = (bb_min[1] + bb_max[1]) / 2
                pcx = (prev_bb_min[0] + prev_bb_max[0]) / 2
                pcy = (prev_bb_min[1] + prev_bb_max[1]) / 2
                xy_dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                if xy_dist > 0.03:
                    issues.append(
                        f"{obj_name} not centered on {prev_name} "
                        f"(offset={xy_dist:.3f}m)")

        if issues:
            return False, "; ".join(issues)
        return True, "Stack order correct"

    def check_goal(self):
        """Check if the stacking goal is achieved."""
        return self.check_stacking_order()

    def get_symbolic_plan(self):
        """Generate symbolic plan template from stacking order."""
        plan = []
        for name in self._stacking_order:
            plan.append(f"pick(['{name}'], {{}})")
            plan.append(
                f"place(['{name}'], {{'x': ?, 'y': ?, 'z': ?, 'theta': ?}})")
        return plan

    def create_task_instances(self, *args, **kwargs):
        pass
