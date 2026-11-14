import logging
import numpy as np

from envs.constants import COLOR_MAP, TAN, BROWN, GREY
from envs.pb_env import PybulletEnv
from utils.io_util import load_json, dump_json
from utils.tamp_util import Action, PrimitiveAction, Parameter
from utils.llm_util import textualize_array
from utils.dynamics_util import apply_configured_dynamics
from utils.pb_util import create_box, set_point

import pybullet as p

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SHELF_PLACE_CLEARANCE = 0.005
SHELF_SURFACE_PENETRATION_TOL = 0.008

# Clearance the hand needs around a shelf placement, measured from the AABB of
# the hand and finger links in the "front" side-grasp orientation: the hand
# spans 0.108 m either side of the tool frame and reaches 0.133 m behind it.
#
# The endpoint maps test only the *object* footprint, so a candidate flush
# against a wall would pass every hard mask with the hand inside the wall.  The
# bounds handed to the planner are therefore inset to where *this gripper* can
# place an object.
GRIPPER_SIDE_CLEARANCE = 0.14
GRIPPER_DEPTH_CLEARANCE = 0.05


class ShelfEnv(PybulletEnv):
    """
    Environment for side-grasp pick-and-place tasks with a multi-level shelf.
    Objects start on the table and must be placed onto designated shelf levels.
    The shelf has multiple horizontal surfaces at different heights.
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
                    "z": Parameter("z", lower_limit=0.0, upper_limit=0.6),
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
        }

    def __str__(self):
        return "ShelfEnv"

    def reset(self, boxes, shelf_config, use_gui=True, **kwargs):
        """Reset environment for a shelf placement task.

        Args:
            boxes: dict of box configs (name, color, w, l, h, x, y, z, ...)
            shelf_config: dict with shelf dimensions and level info:
                {
                    "x": float, "y": float,       # shelf center position
                    "width": float,                # shelf width (x-extent)
                    "depth": float,                # shelf depth (y-extent)
                    # The shelf opens toward -Y for front/side grasps.
                    "levels": [                    # list of shelf levels
                        {"z": float, "label": str, "objects": [str, ...]},
                        ...
                    ],
                    "wall_thickness": float,       # thickness of shelf walls
                }
        """
        super().reset(use_gui=use_gui)

        self._shelf_config = shelf_config
        self._target_placements = {}  # obj_name -> target level index

        # Build target mapping from level config
        for level_idx, level in enumerate(shelf_config["levels"]):
            for obj_name in level.get("objects", []):
                self._target_placements[obj_name] = level_idx

        # Create table
        self.create_table()

        # Create the shelf structure
        self._create_shelf(shelf_config)

        # Create objects (initially on table)
        for box_info in boxes.values():
            roll = box_info.get("roll", 0.0)
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
                    theta=roll,
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

    def _create_shelf(self, cfg):
        """Create a multi-level shelf with horizontal surfaces and vertical side walls."""
        sx = cfg["x"]
        sy = cfg["y"]
        sw = cfg["width"]
        sd = cfg["depth"]
        wt = cfg.get("wall_thickness", 0.01)
        shelf_color = cfg.get("color", [0.55, 0.35, 0.17, 1.0])  # wood brown
        wall_color = [c * 0.9 for c in shelf_color[:3]] + [shelf_color[3]]

        levels = cfg["levels"]
        # Maximum z determines shelf height
        max_z = max(lv["z"] for lv in levels) + 0.15

        # Create shelf surfaces (horizontal boards) for each level
        for i, level in enumerate(levels):
            lz = level["z"]
            board = create_box(w=sw, l=sd, h=wt, color=shelf_color)
            set_point(board, [sx, sy, lz])
            p.changeDynamics(board, -1, mass=0)
            self._objects[f"shelf_board_{i}"] = board

        # Top board (ceiling of shelf)
        top_board = create_box(w=sw, l=sd, h=wt, color=shelf_color)
        set_point(top_board, [sx, sy, max_z])
        p.changeDynamics(top_board, -1, mass=0)
        self._objects["shelf_top"] = top_board

        # Back wall
        back_wall = create_box(w=sw, l=wt, h=max_z, color=wall_color)
        set_point(back_wall, [sx, sy + sd / 2 + wt / 2, max_z / 2])
        p.changeDynamics(back_wall, -1, mass=0)
        self._objects["shelf_back"] = back_wall

        # Left side wall
        left_wall = create_box(w=wt, l=sd, h=max_z, color=wall_color)
        set_point(left_wall, [sx - sw / 2 - wt / 2, sy, max_z / 2])
        p.changeDynamics(left_wall, -1, mass=0)
        self._objects["shelf_wall_left"] = left_wall

        # Right side wall
        right_wall = create_box(w=wt, l=sd, h=max_z, color=wall_color)
        set_point(right_wall, [sx + sw / 2 + wt / 2, sy, max_z / 2])
        p.changeDynamics(right_wall, -1, mass=0)
        self._objects["shelf_wall_right"] = right_wall

    @staticmethod
    def is_shelf_fixture(name):
        return (
            name.startswith("shelf_board_") or
            name in {"shelf_top", "shelf_back", "shelf_wall_left", "shelf_wall_right"}
        )

    def _level_surface_z(self, level_idx):
        wt = self._shelf_config.get("wall_thickness", 0.01)
        return self._shelf_config["levels"][level_idx]["z"] + wt / 2

    def _object_height(self, obj_name):
        bb_min, bb_max = self.get_bb(obj_name)
        return bb_max[2] - bb_min[2]

    def _minimum_center_z_for_level(self, obj_name, level_idx):
        surface_z = self._level_surface_z(level_idx)
        return surface_z + self._object_height(obj_name) / 2 + SHELF_PLACE_CLEARANCE

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
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)

            success, traj, mp_feedback = self.robot.pick(
                obj,
                obstacles,
                grasp_direction="front",
                traj=traj,
                play_traj=play_traj,
                # Lift the newly attached object clear of the table before
                # planning the lateral transfer into the shelf.
                postgrasp_clearance=0.06,
            )
            if success:
                logger.debug("Picked!")
            else:
                logger.debug(f"Pick failed: {mp_feedback}")

        elif action.primitive.name == "place":
            obj_name = action.obj_args[0]
            obj = self.objects[obj_name]
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)
            place_z = action.param_args.get("z", 0.05)
            target_level = self._target_placements.get(obj_name)
            if target_level is not None:
                min_z = self._minimum_center_z_for_level(obj_name, target_level)
                if place_z < min_z:
                    logger.debug(
                        f"Raising shelf place z for {obj_name}: "
                        f"{place_z:.3f} -> {min_z:.3f}"
                    )
                    place_z = min_z

            success, traj, mp_feedback = self.robot.place(
                obj,
                obstacles,
                x=action.param_args["x"],
                y=action.param_args["y"],
                z=place_z,
                theta=action.param_args["theta"],
                traj=traj,
                play_traj=play_traj,
                place_approach="front",
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

    def get_observation(self):
        observation = super().get_observation()

        # Remove non-task objects from observation
        observation.pop("table", None)
        shelf_keys = [k for k in observation if self.is_shelf_fixture(k)]
        for k in shelf_keys:
            observation.pop(k)

        if not observation:
            return observation, ""

        # Shelf info
        cfg = self._shelf_config
        shelf_text = (
            f"There is a shelf at position ({cfg['x']:.2f}, {cfg['y']:.2f}) "
            f"with width {cfg['width']:.2f}m and depth {cfg['depth']:.2f}m. "
            "The shelf opens toward negative y, so place objects near the front half "
            "of the shelf depth when possible."
        )

        # Level descriptions — report board TOP surface z so the LLM can compute
        # correct object-center z as: surface_z + object_height/2.
        level_text = f"The shelf has {len(cfg['levels'])} levels:"
        for i, level in enumerate(cfg['levels']):
            surface_z = self._level_surface_z(i)
            level_text += (
                f"\n  Level {i} ('{level['label']}'): "
                f"surface at z={surface_z:.3f}m, "
                f"target objects: {level.get('objects', [])}"
            )

        # Object info
        names = list(observation.keys())
        obj_text = f"Objects to place on the shelf: {', '.join(names)}."
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

        # Placement status
        status_parts = []
        for obj_name in observation:
            on_shelf, level_idx = self._check_on_shelf(obj_name)
            if on_shelf:
                target_lvl = self._target_placements.get(obj_name)
                correct = (target_lvl is not None and level_idx == target_lvl)
                lvl_label = cfg["levels"][level_idx]["label"]
                status_parts.append(
                    f"{obj_name}: on shelf level {level_idx} ('{lvl_label}')"
                    + (" [CORRECT]" if correct else " [WRONG LEVEL]")
                )
            else:
                status_parts.append(f"{obj_name}: on table (not yet placed on shelf)")

        status_text = "Current status: " + ", ".join(status_parts) + "."

        # Target assignments
        target_text = "Target assignments:"
        for obj_name, lvl_idx in self._target_placements.items():
            lvl_label = cfg["levels"][lvl_idx]["label"]
            surface_z = self._level_surface_z(lvl_idx)
            rec_z = self._minimum_center_z_for_level(obj_name, lvl_idx)
            target_text += (
                f"\n  {obj_name} -> Level {lvl_idx} ('{lvl_label}', "
                f"surface z={surface_z:.3f}m, use center z >= {rec_z:.3f}m)"
            )

        obs_text = "\n".join([shelf_text, level_text, obj_text, status_text, target_text])
        return observation, obs_text

    def _check_on_shelf(self, obj_name, tol=0.03):
        """Check if an object is on any shelf level. Returns (bool, level_index)."""
        if obj_name not in self.objects:
            return False, -1

        bb_min, bb_max = self.get_bb(obj_name)
        pos = self.get_position(obj_name)
        cfg = self._shelf_config

        # Check if within shelf XY bounds
        half_w = cfg["width"] / 2
        half_d = cfg["depth"] / 2
        if (pos[0] < cfg["x"] - half_w - tol or
            pos[0] > cfg["x"] + half_w + tol or
            pos[1] < cfg["y"] - half_d - tol or
            pos[1] > cfg["y"] + half_d + tol):
            return False, -1

        # Check which level the object is on (by z)
        for i, level in enumerate(cfg["levels"]):
            surface_z = self._level_surface_z(i)
            # Object bottom should be near the shelf surface
            obj_bottom = bb_min[2]
            if surface_z - SHELF_SURFACE_PENETRATION_TOL <= obj_bottom <= surface_z + tol:
                return True, i

        return False, -1

    def check_goal(self):
        """Check if all objects are placed on their target shelf levels."""
        issues = []
        cfg = self._shelf_config

        for obj_name, target_lvl in self._target_placements.items():
            if obj_name not in self.objects:
                issues.append(f"{obj_name} not found")
                continue

            on_shelf, actual_lvl = self._check_on_shelf(obj_name)
            if not on_shelf:
                issues.append(f"{obj_name} not on shelf")
            elif actual_lvl != target_lvl:
                target_label = cfg["levels"][target_lvl]["label"]
                actual_label = cfg["levels"][actual_lvl]["label"]
                issues.append(
                    f"{obj_name} on wrong level "
                    f"(on '{actual_label}', should be on '{target_label}')"
                )

            # Check within shelf bounds (not hanging over edge)
            if on_shelf:
                bb_min, bb_max = self.get_bb(obj_name)
                half_w = cfg["width"] / 2
                half_d = cfg["depth"] / 2
                if (bb_min[0] < cfg["x"] - half_w - 0.01 or
                    bb_max[0] > cfg["x"] + half_w + 0.01 or
                    bb_min[1] < cfg["y"] - half_d - 0.01 or
                    bb_max[1] > cfg["y"] + half_d + 0.01):
                    issues.append(f"{obj_name} overhangs shelf edge")

        if issues:
            return False, "; ".join(issues)
        return True, "All objects placed on correct shelf levels"

    def get_symbolic_plan(self):
        """Generate symbolic plan template from target placements."""
        plan = []
        for obj_name in self._target_placements:
            plan.append(f"pick(['{obj_name}'], {{}})")
            plan.append(
                f"place(['{obj_name}'], {{'x': ?, 'y': ?, 'z': ?, 'theta': ?}})")
        return plan

    def get_shelf_bounds(self, level_idx=None, gripper_clearance=True):
        """Get the XY bounds of the shelf, optionally for a specific level.

        Returns dict with x_min, x_max, y_min, y_max, z (surface height).

        By default the bounds are inset by the clearance the hand needs, so
        they describe where this gripper can place an object.  Pass
        ``gripper_clearance=False`` for the raw structural extent.
        """
        cfg = self._shelf_config
        half_w = cfg["width"] / 2
        half_d = cfg["depth"] / 2
        side = GRIPPER_SIDE_CLEARANCE if gripper_clearance else 0.0
        depth = GRIPPER_DEPTH_CLEARANCE if gripper_clearance else 0.0
        if 2 * side >= cfg["width"] or 2 * depth >= cfg["depth"]:
            raise ValueError(
                f"Shelf {cfg['width']:.3f} x {cfg['depth']:.3f} m is too small "
                "to admit the hand; no placement in it can ever succeed"
            )
        bounds = {
            "x_min": cfg["x"] - half_w + side,
            "x_max": cfg["x"] + half_w - side,
            "y_min": cfg["y"] - half_d + depth,
            "y_max": cfg["y"] + half_d - depth,
        }
        if level_idx is not None and level_idx < len(cfg["levels"]):
            bounds["z"] = self._level_surface_z(level_idx)
        return bounds

    def create_task_instances(self, *args, **kwargs):
        pass
