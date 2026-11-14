import logging
import random
import numpy as np
import pybullet as p

from envs.constants import COLOR_MAP
from envs.pb_env import PybulletEnv
from utils.io_util import load_json, dump_json
from utils.tamp_util import Action, PrimitiveAction, Parameter
from utils.llm_util import textualize_array
from utils.dynamics_util import apply_configured_dynamics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PackCompactEnv(PybulletEnv):
    """
    An environment for packing boxes in a compact basket.
    Use top grasp.
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
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
        }

    def __str__(self):
        return "PackCompactEnv"

    def reset(self, basket, boxes, use_gui=True,
              task_type="packing", stacking_order=None, **kwargs):
        super().reset(use_gui=use_gui)

        # Store task configuration for goal checking
        self._task_type = task_type
        self._stacking_order = stacking_order
        self._basket_config = dict(basket)
        self._box_specs = {
            box_info["name"]: dict(box_info)
            for box_info in boxes.values()
            if "name" in box_info
        }

        # Store per-object grasp yaw (rotation around Z at pick time)
        self._grasp_theta_dict = {b["name"]: b.get("grasp_theta", 0.0) for b in boxes.values()}

        # create table
        self.create_table()

        # create basket / walled container
        if basket.get("mesh_basket", False):
            self.create_mesh_basket(
                x=basket["x"], y=basket["y"],
                w=basket["w"], l=basket["l"],
                wall_h=basket.get("wall_h", 0.05),
            )
        elif basket.get("walled", False):
            self.create_walled_container(
                x=basket["x"], y=basket["y"],
                w=basket["w"], l=basket["l"],
                wall_h=basket.get("wall_h", 0.06),
                wall_t=basket.get("wall_t", 0.005),
                floor_includes_walls=basket.get(
                    "floor_includes_walls", True),
            )
        else:
            self.create_basket(x=basket["x"], y=basket["y"], w=basket["w"], l=basket["l"])

        # create boxes
        for box_info in boxes.values():
            ycb_urdf = box_info.get("ycb_urdf")
            init_theta = box_info.get("init_theta", np.pi)
            yaw = box_info.get("yaw", 0.0)
            scaling = box_info.get("global_scaling", 0.07)
            if box_info.get("shape") == "cylinder":
                self.create_customized_cylinder(
                    name=box_info["name"],
                    color=box_info["color"],
                    radius=box_info.get(
                        "radius", box_info["w"] / 2.0),
                    h=box_info["h"],
                    x=box_info["x"],
                    y=box_info["y"],
                    z=box_info["z"],
                )
            elif ycb_urdf:
                loaded = self.create_ycb_object(
                    name=box_info["name"],
                    ycb_urdf=ycb_urdf,
                    x=box_info["x"],
                    y=box_info["y"],
                    z=box_info["z"],
                    theta=init_theta,
                    global_scaling=scaling,
                    yaw=yaw,
                )
                if not loaded:
                    # Fallback to simple box if URDF not found
                    self.create_customized_box(
                        name=box_info["name"],
                        color=box_info["color"],
                        w=box_info["w"],
                        l=box_info["l"],
                        h=box_info["h"],
                        x=box_info["x"],
                        y=box_info["y"],
                        z=box_info["z"],
                        theta=init_theta,
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
                    theta=init_theta,
                    yaw=yaw,
                )
            apply_configured_dynamics(
                self._objects[box_info["name"]],
                box_info,
                pybullet_module=p,
            )
        # physical simulation
        self.simulate()

        observation = self.get_observation()
        return observation

    def apply_action(self, action: Action, play_traj: bool = False):
        # sanity check
        if action is None:
            return False, "No action is given!"
        if not action.primitive in self._primitive_actions.values():
            return False, "Unknown primitive action!"
        for obj_name in action.obj_args:
            if not obj_name in self.objects:
                return False, "Unknown object name!"

        if action.traj is not None and len(action.traj) > 0:
            traj = action.traj
        else:
            traj = None

        if action.primitive.name == "pick":
            obj_name = action.obj_args[0]
            object = self.objects[obj_name]

            # prepare obstacles (avoid all other objects)
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)

            grasp_yaw = self._grasp_theta_dict.get(obj_name, 0.0)
            object_cfg = self._box_specs.get(obj_name, {})
            success, traj, mp_feedback = self.robot.pick(
                object, obstacles, grasp_direction="top", traj=traj,
                play_traj=play_traj, grasp_yaw=grasp_yaw,
                grasp_z_offset=object_cfg.get("grasp_z_offset"),
                pregrasp_clearance=object_cfg.get(
                    "pregrasp_clearance", 0.08),
            )
            if success:
                logger.debug("Picked!")
            else:
                logger.debug(f"Pick is not executed:{mp_feedback}")

            # don't simulate at pick

        elif action.primitive.name == "place":
            obj_name = action.obj_args[0]
            object = self.objects[obj_name]

            # Compute place_z from the object's actual half-height above the basket
            # floor. Releasing small objects too high makes them fall and slide,
            # which rehearsal reports as unstable placement drift.
            if "z" in action.param_args:
                place_z = action.param_args["z"]
            else:
                bb_min, bb_max = self.get_bb(obj_name)
                obj_half_height = (bb_max[2] - bb_min[2]) / 2.0
                BASKET_FLOOR_Z = 0.010  # mesh basket z=0.01 (floor sits at body z)
                place_z = BASKET_FLOOR_Z + obj_half_height

            # prepare obstacles (avoid all other objects)
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)

            success, traj, mp_feedback = self.robot.place(
                object,
                obstacles,
                # randomly sample x,y for ablation study
                x=action.param_args["x"],
                y=action.param_args["y"],
                z=place_z,
                theta=action.param_args["theta"],
                traj=traj,
                play_traj=play_traj,
            )
            # only simulate at successful place
            if success:
                self.simulate()
                self.theta_dict[obj_name] = action.param_args["theta"]
                logger.debug("Placed!")
            else:
                logger.debug(f"Place is not executed:{mp_feedback}")

        # assign traj
        if action.traj is None or len(action.traj) == 0:
            action.traj = traj

        return success, mp_feedback

    def get_observation(self):
        observation = super().get_observation()
        # remove table & basket (+ walls) from observation
        observation.pop("table", None)
        basket_obs = observation.pop("basket", None)
        # Remove wall entries from walled containers
        wall_keys = [k for k in observation if k.startswith("basket_wall_")]
        for k in wall_keys:
            observation.pop(k)

        if basket_obs is None:
            return observation, ""

        # textualize observation
        # add basket info
        x_range = np.array([basket_obs["bb_min"][0], basket_obs["bb_max"][0]])
        y_range = np.array([basket_obs["bb_min"][1], basket_obs["bb_max"][1]])
        basket_text = f"The basket has a rectangular shape, ranges {textualize_array(x_range)} along the x axis, and ranges {textualize_array(y_range)} along the y axis."

        # add box info
        boxes_text = f"There are several boxes in the envrionment: {', '.join(observation.keys())}."
        for object_name, object_state in observation.items():
            boxes_text += f"\n{object_name} is at position {textualize_array(object_state['position'])}, and it has min bounding box corner {textualize_array(object_state['bb_min'])} and max bounding box corner {textualize_array(object_state['bb_max'])},"
            width = object_state["bb_max"][0] - object_state["bb_min"][0]
            length = object_state["bb_max"][1] - object_state["bb_min"][1]
            height = object_state["bb_max"][2] - object_state["bb_min"][2]
            boxes_text += f"its length along x axis is {textualize_array(width)}, its length along y axis is {textualize_array(length)}, its height is {textualize_array(height)}."

        # predicate info
        predicate_list = []
        for object_name in observation.keys():
            if self.check_in_basket(object_name):
                predicate_list.append(f"{object_name} is in basket")
            else:
                predicate_list.append(f"{object_name} is not in basket")

        predicate_text = ", ".join(predicate_list) + "."

        obs_text = basket_text + "\n" + boxes_text + "\n" + predicate_text
        return observation, obs_text

    def get_symbolic_plan(self):
        return [
            "pick(['red_box'], {})",
            "place(['red_box'], {'x': ?, 'y': ?, 'theta': ?})",
            "pick(['blue_box'], {})",
            "place(['blue_box'], {'x': ?, 'y': ?, 'theta': ?})",
            "pick(['cyan_box'], {})",
            "place(['cyan_box'], {'x': ?, 'y': ?, 'theta': ?})",
            "pick(['yellow_box'], {})",
            "place(['yellow_box'], {'x': ?, 'y': ?, 'theta': ?})",
        ]

    def check_in_basket(self, obj_name, tol=0.015):
        min_bb_basket, max_bb_basket = self.get_bb("basket")
        min_bb_obj, max_bb_obj = self.get_bb(obj_name)

        if (
            min_bb_obj[0] > min_bb_basket[0] - tol
            and max_bb_obj[0] < max_bb_basket[0] + tol
            and min_bb_obj[1] > min_bb_basket[1] - tol
            and max_bb_obj[1] < max_bb_basket[1] + tol
        ):
            return True
        return False

    def check_stacking_order(self, stacking_order, tol=0.02):
        """Check if objects are stacked in the specified order (bottom to top).
        
        Args:
            stacking_order: list of object names from bottom to top
            tol: tolerance for z-alignment and xy-overlap
            
        Returns:
            (bool, str): (success, feedback message)
        """
        issues = []
        for i in range(len(stacking_order)):
            obj_name = stacking_order[i]
            if obj_name not in self.objects or obj_name in ("basket", "table"):
                issues.append(f"{obj_name} not found")
                continue
            
            bb_min, bb_max = self.get_bb(obj_name)
            
            if i == 0:
                # Bottom object should be on the floor/basket
                if bb_min[2] > 0.03:
                    issues.append(f"{obj_name} (bottom) not on surface (z_min={bb_min[2]:.3f})")
            else:
                # Each subsequent object should be above the previous
                prev_name = stacking_order[i - 1]
                if prev_name not in self.objects:
                    continue
                prev_bb_min, prev_bb_max = self.get_bb(prev_name)
                
                # Check z: this object's bottom should be near previous object's top
                z_gap = bb_min[2] - prev_bb_max[2]
                if z_gap > tol:
                    issues.append(f"{obj_name} not on top of {prev_name} (gap={z_gap:.3f}m)")
                elif z_gap < -tol:
                    issues.append(f"{obj_name} intersects {prev_name}")
                
                # Check xy overlap: should be centered above
                cx = (bb_min[0] + bb_max[0]) / 2
                cy = (bb_min[1] + bb_max[1]) / 2
                pcx = (prev_bb_min[0] + prev_bb_max[0]) / 2
                pcy = (prev_bb_min[1] + prev_bb_max[1]) / 2
                xy_dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                if xy_dist > 0.03:
                    issues.append(f"{obj_name} not centered on {prev_name} (offset={xy_dist:.3f}m)")

        if issues:
            return False, "; ".join(issues)
        return True, "Stack order correct"

    def check_goal(self, task_type=None, stacking_order=None):
        """Check goal based on task type.
        
        Args:
            task_type: 'packing' (all in basket) or 'stacking' (correct stack order).
                       Defaults to value set during reset().
            stacking_order: required for stacking tasks, list of names bottom-to-top.
                            Defaults to value set during reset().
        """
        task_type = task_type or getattr(self, "_task_type", "packing")
        stacking_order = stacking_order or getattr(self, "_stacking_order", None)
        if task_type == "stacking" and stacking_order:
            return self.check_stacking_order(stacking_order)

        # Default: packing (all objects in basket)
        is_goal = True
        feedback = []
        for obj_name in self.objects.keys():
            if obj_name != "basket" and obj_name != "table" and not obj_name.startswith("basket_wall_"):
                if not self.check_in_basket(obj_name):
                    is_goal = False
                    feedback.append(f"{obj_name} is not in basket")

        return is_goal, ", ".join(feedback)

    def create_task_instances(
        self,
        env_config,
        num_instances,
        save_to_file=False,
        instance_file=None,
        overwrite=False,
    ):
        # if already exists, load from file
        if instance_file is not None and instance_file.exists() and not overwrite:
            task_instances = load_json(instance_file)
            logger.info(f"Load from existing file {instance_file}.")

        else:
            task_instances = {}
            for task_i in range(num_instances):
                basket_info = {
                    "x": env_config.basket.x,
                    "y": env_config.basket.y,
                    "w": env_config.basket.w,
                    "l": env_config.basket.l,
                }

                # sample box locations
                # todo: fill in algorithm for sampling initial box locations
                boxes_info = {}
                xmin, xmax = [0.385, 0.395]
                last_y = -0.6
                gripper_length = 0.09
                for box_i, (box_name, box) in enumerate(env_config.boxes.items()):
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(
                        max(-0.6 + 0.3 * box_i, last_y + box.l / 2),
                        -0.6 + 0.3 * (box_i + 1),
                    )
                    y = max(y, last_y + box.l / 2 + gripper_length)
                    last_y = y + box.l / 2

                    box_info = {
                        "name": box_name,
                        "color": COLOR_MAP[box.color],
                        "w": box.w,
                        "l": box.l,
                        "h": box.h,
                        "x": x,
                        "y": y,
                        "z": 0.06,
                    }

                    boxes_info[box_i] = box_info

                instance = {"basket": basket_info, "boxes": boxes_info}
                task_instances[task_i] = instance

            if save_to_file:
                assert (
                    instance_file is not None
                ), "instance_file must be specified when save_to_file is True"

                dump_json(task_instances, instance_file)

        return task_instances
