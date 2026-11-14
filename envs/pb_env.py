import os
import abc
import copy
import logging
import numpy as np
from scipy.spatial.transform import Rotation

import pybullet as p
from pybullet_planning import (
    connect,
    disconnect,
    is_connected,
    create_obj,
    set_pose,
    Pose,
    load_pybullet,
    get_movable_joints,
    link_from_name,
    set_joint_positions,
    get_joint_positions,
    create_attachment,
    get_link_pose,
    get_distance,
    quat_angle_between,
    wait_for_duration,
    multiply,
    LockRenderer,
)
from pybullet_planning.interfaces.robots.collision import get_collision_fn

from utils.planning_util import (
    check_ee_collision,
    densify_joint_path,
    plan_joint_motion,
    time_parameterize_joint_path,
)
from utils import replay_capture
from utils.pb_util import (
    assign_link_colors,
    create_box,
    create_cylinder,
    set_point,
    get_pose,
    set_static,
    add_data_path,
)
from utils.tamp_util import Action, CachedTrajectory, TAMPFeedback
from utils.metrics_util import (
    current_phase,
    increment_metric,
    metric_timer,
    record_metric_event,
)
from envs.constants import ASSETS_DIR, COLOR_FRANKA, BROWN, TAN

logger = logging.getLogger(__name__)


BOX_PATH = os.path.join(ASSETS_DIR, "box_obstacle3.obj")
HOOK_PATH = os.path.join(ASSETS_DIR, "hook.obj")
FRANKA_ROBOT_URDF = os.path.join(
    ASSETS_DIR,
    "franka_description_official",
    "robots",
    "fr3_franka_hand.urdf",
)
FR3_ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
TRAJECTORY_TIME_STEP = 1.0 / 60.0
TRAJECTORY_MAX_JOINT_VELOCITY = 0.5
TRAJECTORY_MAX_JOINT_ACCELERATION = 2.0
# PyBullet's constrained IK can return a value a few 1e-4 radians beyond the
# requested interval.  Such numerical spill is projected back onto the
# already-conservative planning bound and then subjected to the usual FK and
# collision checks.  Larger violations remain hard failures.
IK_NUMERICAL_LIMIT_TOLERANCE = 1e-3
YCB_ASSETS_DIR = os.path.join(ASSETS_DIR, "ycb_objects")


class PybulletEnv:
    def __init__(self):
        self._objects = {}
        # Visual room fixtures are deliberately kept separate from task
        # objects.  They should appear in every GUI scene without being sent
        # to the LLM as manipulable objects or changing task goal checks.
        self._background_bodies = {}
        self._primitive_actions = {}
        self.theta_dict = {}

    @property
    def objects(self):
        return self._objects

    @property
    def primitive_actions(self):
        return self._primitive_actions

    @property
    def body_id_name_mapping(self):
        return {v: k for k, v in self._objects.items()}

    def reset(self, use_gui=True, load_floor=True):
        floor_body = None
        if is_connected():
            # resetSimulation removes every body (including the plane) without
            # touching the GUI connection; incremental removeBody calls can
            # corrupt body IDs and leave a blank window.
            p.resetSimulation()
            p.setGravity(0, 0, -10)
            add_data_path()
            if load_floor:
                floor_body = p.loadURDF("plane.urdf", [0, 0, self.FLOOR_Z])
        else:
            # First time: create the connection
            connect(use_gui=use_gui)
            p.setGravity(0, 0, -10)
            add_data_path()
            if load_floor:
                floor_body = p.loadURDF("plane.urdf", [0, 0, self.FLOOR_Z])

        if floor_body is not None:
            # plane.urdf carries a checker/grid texture in some PyBullet data
            # distributions that changeVisualShape cannot reliably clear, so
            # hide the imported plane and lay a flat slab over it.
            p.changeVisualShape(floor_body, -1, rgbaColor=(0, 0, 0, 0))
            studio_floor = create_box(
                w=self.FLOOR_SIZE, l=self.FLOOR_SIZE,
                h=self.FLOOR_SLAB_THICKNESS, color=self.FLOOR_COLOR)
            set_point(studio_floor, [
                self.TABLE_CENTER_X, self.TABLE_CENTER_Y,
                self.FLOOR_Z - self.FLOOR_SLAB_THICKNESS / 2.0])
            set_static(studio_floor)

        # reset objects
        self._objects = {}
        self._background_bodies = {}
        self.theta_dict = {}

        # add pybullet robot
        self.robot = PyBulletRobot()

        # Use the same linked rear/left studio walls in every environment.
        # Their placement is derived from the shared table geometry and is
        # therefore consistent across packing, pushing, shelf, stacking and
        # Tower-of-Hanoi demos.
        self.create_studio_backdrop()

        # reset camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=40,
            cameraPitch=-45,
            cameraTargetPosition=[self.TABLE_CENTER_X, self.TABLE_CENTER_Y, 0.0],
        )
        try:
            p.configureDebugVisualizer(lightPosition=list(self.LIGHT_POSITION))
        except Exception as exc:  # noqa: BLE001 - lighting is never fatal
            logger.debug("could not place the light: %s", exc)

        # The scene is now built. Nothing listens unless a caller asks to, and
        # the benchmark never does; a video recorder does, because the
        # resetSimulation above silently terminates any log already running.
        replay_capture.scene_ready()

    def destroy(self):
        if is_connected():
            disconnect()

    def simulate(self, num_steps=100):
        increment_metric("simulator_steps", num_steps)
        for _ in range(num_steps):
            p.stepSimulation()

    def collision_function(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        if xmin1 < xmax2 and xmax1 > xmin2 and ymin1 < ymax2 and ymax1 > ymin2:
            return True
        return False

    def step(self, action: Action, *args, **kwargs):
        # Some callers store only ``Action.traj``; shadow-rollout callers also
        # keep phase boundaries so pick attachment and place release happen at
        # the exact validated waypoint.  Wrap the list here so both work.
        if (
            getattr(action, "traj", None)
            and getattr(action, "motion_trace", None)
            and not isinstance(action.traj, CachedTrajectory)
        ):
            action.traj = CachedTrajectory(
                action.traj, copy.deepcopy(action.motion_trace))

        # apply action
        increment_metric("simulator_primitive_attempts")
        action_name = getattr(getattr(action, "primitive", None), "name", None)
        record_metric_event(
            "simulator_primitive_started",
            action=action_name,
            action_text=str(action),
        )
        with metric_timer("simulator_primitive"):
            success, mp_feedback = self.apply_action(action, *args, **kwargs)
        if success:
            increment_metric("simulator_primitive_successes")
            robot_trace = copy.deepcopy(
                getattr(getattr(self, "robot", None), "last_motion_trace", [])
            )
            action.motion_trace = robot_trace
            if getattr(action, "traj", None):
                action.traj = CachedTrajectory(action.traj, robot_trace)
        else:
            increment_metric("simulator_primitive_failures")

        # get observation
        observation = self.get_observation()

        # check goal
        goal_achieved, goal_feedback = self.check_goal()

        # prepare feedback
        feedback = TAMPFeedback(
            motion_planner_feedback=mp_feedback,
            task_process_feedback=goal_feedback,
            action_success=success,
            goal_achieved=goal_achieved,
        )
        record_metric_event(
            "simulator_primitive_finished",
            action=action_name,
            action_text=str(action),
            action_success=bool(success),
            goal_achieved=bool(goal_achieved),
            feedback=mp_feedback,
            metric_phase=current_phase(),
        )
        logger.info(feedback.motion_planner_feedback)

        if goal_achieved:
            logger.debug("Goal achieved!")

        return observation, feedback

    def get_observation(self):
        # in this general function, we assume getting observations for every objects
        # cprint("here is in the observation function", "yellow")
        observation = {}
        for obj_name, obj in self.objects.items():
            # for position
            pos = self.get_position(obj_name)
            # bbox = self.get_bounding_box(obj_name)
            bb_min, bb_max = self.get_bb(obj_name)

            observation[obj_name] = {"position": pos, "bb_min": bb_min, "bb_max": bb_max}
        return observation

    @abc.abstractmethod
    def apply_action(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def check_goal(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def create_task_instances(self):
        raise NotImplementedError("Override me!")

    # Real tabletop layout measured in the robot base frame.
    # Units are meters. The Franka base is the world origin; +x points from the
    # robot toward the front edge / task area and +y runs along the table length.
    TABLE_LENGTH_Y = 1.23
    TABLE_WIDTH_X = 1.00
    ROBOT_TO_FRONT_EDGE_X = 0.70
    ROBOT_TO_BACK_EDGE_X = TABLE_WIDTH_X - ROBOT_TO_FRONT_EDGE_X
    TABLE_CENTER_X = (ROBOT_TO_FRONT_EDGE_X - ROBOT_TO_BACK_EDGE_X) / 2.0
    TABLE_CENTER_Y = 0.0
    TABLE_TOP_Z = 0.0
    TABLE_HEIGHT = 1.0
    FLOOR_Z = TABLE_TOP_Z - TABLE_HEIGHT
    TABLE_THICKNESS = 0.04
    TABLE_LEG_WIDTH = 0.045
    TABLE_LEG_INSET = 0.07
    TABLE_COLOR = (0.84, 0.85, 0.82, 1.0)
    TABLE_LEG_COLOR = (0.72, 0.73, 0.70, 1.0)
    FLOOR_COLOR = (0.72, 0.73, 0.72, 1.0)
    # Our own floor slab, laid over the imported plane so no mesh texture can
    # reach the picture. Wide enough that no demo camera sees past its edge.
    FLOOR_SIZE = 8.0
    FLOOR_SLAB_THICKNESS = 0.02
    # PyBullet's default light leaves the studio flat and grey. Placing it
    # above and in front of the table lifts the whole scene and gives the arm
    # a shadow, which is most of what makes a recording readable.
    LIGHT_POSITION = (1.0, -0.9, 2.2)
    WALL_COLOR = (0.96, 0.96, 0.96, 1.0)
    WALL_THICKNESS = 0.02
    WALL_TABLE_CLEARANCE = 0.60
    WALL_TOP_Z = TABLE_TOP_Z + 1.68

    def create_studio_backdrop(self):
        """Create the linked back and left walls used by all demo scenes.

        Both walls start at floor level, stand 0.60 m beyond the corresponding
        table edge, and overlap at their corner.  Background bodies are not
        registered as task objects, so they cannot
        leak into observations, LLM prompts, or object-level goal checks.
        """
        if self._background_bodies:
            return dict(self._background_bodies)

        wall_bottom_z = self.FLOOR_Z
        wall_height = self.WALL_TOP_Z - wall_bottom_z
        wall_center_z = (wall_bottom_z + self.WALL_TOP_Z) / 2.0
        wall_t = self.WALL_THICKNESS
        clearance = self.WALL_TABLE_CLEARANCE

        table_x_min = self.TABLE_CENTER_X - self.TABLE_WIDTH_X / 2.0
        table_x_max = self.TABLE_CENTER_X + self.TABLE_WIDTH_X / 2.0
        table_y_min = self.TABLE_CENTER_Y - self.TABLE_LENGTH_Y / 2.0
        table_y_max = self.TABLE_CENTER_Y + self.TABLE_LENGTH_Y / 2.0

        back_wall_x = table_x_min - clearance - wall_t / 2.0
        left_wall_y = table_y_max + clearance + wall_t / 2.0

        # Join at the rear-left corner.  Double the rear wall toward -Y and the
        # left wall toward +X while keeping 0.60 m clearance from the table.
        back_wall_y_max = left_wall_y + wall_t / 2.0
        back_wall_y_min = table_y_min - clearance
        back_wall_length = 2.0 * (back_wall_y_max - back_wall_y_min)
        back_wall_y_min = back_wall_y_max - back_wall_length
        back_wall_y = (back_wall_y_min + back_wall_y_max) / 2.0

        left_wall_x_min = back_wall_x - wall_t / 2.0
        left_wall_x_max = table_x_max + clearance
        left_wall_width = 2.0 * (left_wall_x_max - left_wall_x_min)
        left_wall_x = left_wall_x_min + left_wall_width / 2.0

        back_wall = create_box(
            w=wall_t,
            l=back_wall_length,
            h=wall_height,
            mass=0,
            color=self.WALL_COLOR,
        )
        set_point(back_wall, [back_wall_x, back_wall_y, wall_center_z])

        left_wall = create_box(
            w=left_wall_width,
            l=wall_t,
            h=wall_height,
            mass=0,
            color=self.WALL_COLOR,
        )
        set_point(left_wall, [left_wall_x, left_wall_y, wall_center_z])

        self._background_bodies = {
            "back_wall": back_wall,
            "left_wall": left_wall,
        }
        return dict(self._background_bodies)

    def create_table(self, name="table", color=None, x=None, y=None, z=None):
        if color is None:
            color = self.TABLE_COLOR
        if x is None:
            x = self.TABLE_CENTER_X
        if y is None:
            y = self.TABLE_CENTER_Y
        if z is None:
            z = self.TABLE_TOP_Z

        tabletop_center_z = z - self.TABLE_THICKNESS / 2.0
        table_body = create_box(
            w=self.TABLE_WIDTH_X,
            l=self.TABLE_LENGTH_Y,
            h=self.TABLE_THICKNESS,
            color=color,
        )
        set_point(table_body, [x, y, tabletop_center_z])
        self._objects[name] = table_body

        leg_height = z - self.TABLE_THICKNESS - self.FLOOR_Z
        leg_center_z = self.FLOOR_Z + leg_height / 2.0
        x_min = x - self.TABLE_WIDTH_X / 2.0 + self.TABLE_LEG_INSET
        x_max = x + self.TABLE_WIDTH_X / 2.0 - self.TABLE_LEG_INSET
        y_min = y - self.TABLE_LENGTH_Y / 2.0 + self.TABLE_LEG_INSET
        y_max = y + self.TABLE_LENGTH_Y / 2.0 - self.TABLE_LEG_INSET
        leg_positions = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
            (x, y),
        ]
        for lx, ly in leg_positions:
            leg = create_box(
                w=self.TABLE_LEG_WIDTH,
                l=self.TABLE_LEG_WIDTH,
                h=leg_height,
                color=self.TABLE_LEG_COLOR,
            )
            set_point(leg, [lx, ly, leg_center_z])

    def create_basket(self, name="basket", color=BROWN, x=0.6, y=0, z=0.002, w=0.2, l=0.4, h=0.01):
        basket_body = create_box(w=w, l=l, h=h, color=color)
        set_point(basket_body, [x, y, z])
        self._objects[name] = basket_body

    # Natural mesh extents as loaded by PyBullet at scale=(1,1,1) for model.obj
    # Measured via p.getAABB() with collision=mesh on the exported OBJ file.
    # AABB at scale=1: X=[42.628,59.982] Y=[49.970,81.669] Z=[-0.003,9.946]
    _BASKET_MESH_RAW_W  = 17.354  # X span at scale=1 (raw OBJ units ≈ inches)
    _BASKET_MESH_RAW_L  = 31.699  # Y span at scale=1
    _BASKET_MESH_RAW_H  =  9.949  # Z span at scale=1 (floor → rim)
    _BASKET_MESH_CX     = 51.305  # X centroid at scale=1
    _BASKET_MESH_CY     = 65.820  # Y centroid at scale=1
    _BASKET_MESH_ZFLOOR = -0.003  # Z floor at scale=1

    def create_mesh_basket(self, name="basket", x=0.65, y=0, z=0.01,
                           w=0.32, l=0.21, wall_h=0.05):
        """Load the realistic basket mesh via a dynamically generated URDF.

        A URDF is written to assets/basket/basket.urdf with the correct
        non-uniform mesh scale (sx, sy, sz) and a 180° X-rotation so the
        basket is right-side up.  A flat box collision shape inside the URDF
        provides the AABB used by check_in_basket().  Invisible physics walls
        are added separately so objects cannot fall out.
        """
        basket_dir = os.path.join(ASSETS_DIR, "basket")
        urdf_path  = os.path.join(basket_dir, "basket.urdf")

        # Non-uniform scale: fit raw extents to desired physical dimensions
        sx = w / self._BASKET_MESH_RAW_W
        sy = l / self._BASKET_MESH_RAW_L
        sz = wall_h / self._BASKET_MESH_RAW_H

        # Centre the mesh over the body origin and align its floor to body z.
        # The raw mesh is Z_UP (already upright — no rotation needed).
        ox = -self._BASKET_MESH_CX     * sx   # shift X centroid to 0
        oy = -self._BASKET_MESH_CY     * sy   # shift Y centroid to 0
        oz = -self._BASKET_MESH_ZFLOOR * sz   # shift floor to z=0

        urdf_content = f"""<?xml version="1.0" ?>
<robot name="basket">
  <link name="base_link">
    <visual>
      <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0" />
      <geometry>
        <mesh filename="model.obj" scale="{sx:.8f} {sy:.8f} {sz:.8f}" />
      </geometry>
      <material name="basket_color">
        <color rgba="0.62 0.62 0.62 1.0" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="{w:.6f} {l:.6f} 0.006" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0" />
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />
    </inertial>
  </link>
</robot>
"""
        with open(urdf_path, "w") as fh:
            fh.write(urdf_content)

        # PyBullet resolves mesh paths relative to the URDF file location
        p.setAdditionalSearchPath(basket_dir)
        basket_id = p.loadURDF(urdf_path, [x, y, z], useFixedBase=True)
        # PyBullet ignores URDF <color> tags on COLLADA meshes — force the
        # colour on every visual shape link so the basket isn't rendered black.
        p.changeVisualShape(basket_id, -1, rgbaColor=[0.62, 0.62, 0.62, 1.0])
        self._objects[name] = basket_id

        # Invisible physics walls so objects stay inside
        wall_z = z + wall_h / 2
        invis = [0.0, 0.0, 0.0, 0.0]
        wall_t = 0.005
        for sign, label, (bw, bl, bh) in [
            (+1, "px", (wall_t, l, wall_h)),
            (-1, "mx", (wall_t, l, wall_h)),
            (+1, "py", (w, wall_t, wall_h)),
            (-1, "my", (w, wall_t, wall_h)),
        ]:
            offset_x = sign * (w / 2 + wall_t / 2) if "x" in label else 0
            offset_y = sign * (l / 2 + wall_t / 2) if "y" in label else 0
            wall = create_box(w=bw, l=bl, h=bh, color=invis)
            set_point(wall, [x + offset_x, y + offset_y, wall_z])
            p.changeDynamics(wall, -1, mass=0)
            self._objects[f"{name}_wall_{label}"] = wall

    def create_walled_container(
        self,
        name="basket",
        color=BROWN,
        x=0.6,
        y=0,
        z=0.002,
        w=0.2,
        l=0.4,
        wall_h=0.06,
        wall_t=0.005,
        floor_includes_walls=True,
    ):
        """Create a container with floor + 4 walls using PyBullet collision shapes.
        
        Args:
            w, l: inner dimensions of the container
            wall_h: wall height
            wall_t: wall thickness
            floor_includes_walls: when true, the floor spans the outer wall
                footprint; when false, the basket body's AABB represents the usable
                inner packing area while the walls still define the requested
                external dimensions.
        """
        # Floor
        floor_w = w + 2 * wall_t if floor_includes_walls else w
        floor_l = l + 2 * wall_t if floor_includes_walls else l
        floor_body = create_box(
            w=floor_w, l=floor_l, h=0.005, color=color)
        set_point(floor_body, [x, y, z])
        p.changeDynamics(floor_body, -1, mass=0)
        self._objects[name] = floor_body

        # 4 walls (static)
        wall_color = [c * 0.8 for c in color[:3]] + [color[3] if len(color) > 3 else 0.7]
        wall_z = z + wall_h / 2 + 0.0025  # half wall above floor

        # Wall +x
        wx = create_box(w=wall_t, l=l + 2 * wall_t, h=wall_h, color=wall_color)
        set_point(wx, [x + w / 2 + wall_t / 2, y, wall_z])
        p.changeDynamics(wx, -1, mass=0)
        self._objects[f"{name}_wall_px"] = wx

        # Wall -x
        wmx = create_box(w=wall_t, l=l + 2 * wall_t, h=wall_h, color=wall_color)
        set_point(wmx, [x - w / 2 - wall_t / 2, y, wall_z])
        p.changeDynamics(wmx, -1, mass=0)
        self._objects[f"{name}_wall_mx"] = wmx

        # Wall +y
        wy = create_box(w=w, l=wall_t, h=wall_h, color=wall_color)
        set_point(wy, [x, y + l / 2 + wall_t / 2, wall_z])
        p.changeDynamics(wy, -1, mass=0)
        self._objects[f"{name}_wall_py"] = wy

        # Wall -y
        wmy = create_box(w=w, l=wall_t, h=wall_h, color=wall_color)
        set_point(wmy, [x, y - l / 2 - wall_t / 2, wall_z])
        p.changeDynamics(wmy, -1, mass=0)
        self._objects[f"{name}_wall_my"] = wmy

    def create_customized_box(
        self,
        name,
        color,
        w,
        l,
        h,
        x,
        y,
        z,
        theta=np.pi,
        yaw=0.0,
        pitch=0.0,
    ):
        body = create_box(w=w, l=l, h=h, color=color, mass=1.0)
        set_pose(
            body,
            Pose(point=[x, y, z], euler=np.array([theta, pitch, yaw])),
        )
        self._objects[name] = body
        self.theta_dict[name] = 0

    def create_customized_cylinder(self, name, color, radius, h, x, y, z):
        body = create_cylinder(radius=radius, height=h, color=color, mass=1.0)
        set_pose(body, Pose(point=[x, y, z]))
        self._objects[name] = body
        self.theta_dict[name] = 0

    def create_ycb_object(self, name, ycb_urdf, x, y, z, theta=np.pi, global_scaling=0.065, yaw=0.0):
        """Load a YCB object from its URDF file instead of a simple box primitive."""
        urdf_path = os.path.join(YCB_ASSETS_DIR, "ycb_assets", ycb_urdf)
        if not os.path.exists(urdf_path):
            logger.warning(f"YCB URDF not found: {urdf_path}, falling back to box")
            return False
        p.setAdditionalSearchPath(os.path.join(YCB_ASSETS_DIR, "ycb_assets"))
        orientation = p.getQuaternionFromEuler([theta, 0, yaw])
        body = p.loadURDF(urdf_path, [x, y, z], orientation,
                          useFixedBase=False, globalScaling=global_scaling)
        p.changeDynamics(body, -1, mass=1.0)
        self._objects[name] = body
        self.theta_dict[name] = 0
        return True

    def create_box(self, name, color, x, y, z, theta=np.pi):
        box_body = create_obj(BOX_PATH, scale=0.5, color=color, mass=1.0)
        set_pose(box_body, Pose(point=[x, y, z], euler=np.array([theta, 0, 0])))
        self._objects[name] = box_body
        self.theta_dict[name] = 0

    def create_tool(self, name, color, x, y, z, theta=np.pi):
        tool_body = create_obj(HOOK_PATH, scale=0.3, color=color, mass=1.0)
        set_pose(tool_body, Pose(point=[x, y, z], euler=np.array([theta, 0, 0])))
        self._objects[name] = tool_body

    def get_position(self, name):
        return get_pose(self._objects[name])[0]

    def get_orientation(self, name):
        return get_pose(self._objects[name])[1]

    def get_bb(self, name):
        return p.getAABB(self._objects[name])

    def prepare_obstacles(self, obj_name_list=[], remove_mode=False):
        obstacles = {}
        # if remove mode, remove objects in obj_name_list
        if remove_mode:
            for obj_name, id in self._objects.items():
                if obj_name not in obj_name_list:
                    obstacles[id] = obj_name
        else:
            # add obstacles
            for obj_name in obj_name_list:
                obstacles[self._objects[obj_name]] = obj_name

        return obstacles


# Post-place withdrawal schedules.  The first entry is the preferred
# distance; the shorter fallbacks exist because a bay-confined placement can
# leave the hand in a narrow passage that a single query fails to solve.
RETREAT_DISTANCE_SCHEDULE = (0.12, 0.09, 0.06, 0.04)
LIFT_HEIGHT_SCHEDULE = (0.10, 0.07, 0.05, 0.03)


class PyBulletRobot:
    def __init__(self):
        # Load the official Franka Research 3 model.  Planning limits are read
        # from this same URDF below, so the collision model, IK, and sampler
        # cannot silently disagree about which robot is being planned for.
        self.urdf_path = FRANKA_ROBOT_URDF
        self.robot = load_pybullet(self.urdf_path, fixed_base=True)
        assign_link_colors(self.robot, colors=COLOR_FRANKA)
        self.ik_joints = get_movable_joints(self.robot)
        self.arm_joints = [
            joint
            for joint in self.ik_joints
            if p.getJointInfo(self.robot, joint)[1].decode("utf-8")
            in FR3_ARM_JOINT_NAMES
        ]
        if len(self.arm_joints) != 7:
            raise RuntimeError(
                "The loaded FR3 description does not expose exactly seven arm joints"
            )

        # A small numerical margin keeps sampling and IK away from the exact
        # hardware position stops. The underlying values still come directly
        # from the official URDF.
        joint_limit_margin = 1e-4
        self.planning_limits = {}
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        for joint in self.ik_joints:
            joint_info = p.getJointInfo(self.robot, joint)
            lower, upper = float(joint_info[8]), float(joint_info[9])
            self.ik_lower_limits.append(lower)
            self.ik_upper_limits.append(upper)
            if joint in self.arm_joints:
                self.planning_limits[joint] = (
                    lower + joint_limit_margin,
                    upper - joint_limit_margin,
                )
        self.ik_joint_ranges = [
            upper - lower
            for lower, upper in zip(self.ik_lower_limits, self.ik_upper_limits)
        ]
        self.tool_attach_link = link_from_name(self.robot, "fr3_hand_tcp")
        self.trajectory_time_step = TRAJECTORY_TIME_STEP

        # set static: important that the robot is static during planning
        set_static(self.robot)

        # grasping attachments & direction
        self.attachments_robot = []
        self.last_grasp_direction = None
        self.last_grasp_position_offset = None
        self.last_motion_trace = []
        self.motion_planner_call_count = 0

        # set grasping offset: position and orientation
        self.position_offset_dict = {
            "top": np.array([0.0, 0, 0.05]),
            "left": np.array([0.5, -0.15, 0]),
            "right": np.array([0, 0.08, 0]),
            "forward": np.array([-0.1, 0, 0]),
            # "front": approach from the -Y direction (gripper points in +Y).
            # Used for shelf placement where the shelf opens on its -Y face.
            # Keep the object tucked between the fingers while leaving a small
            # front standoff and table clearance for side grasps.
            "front": np.array([0, -0.035, 0.02]),
        }

        x_axis_positive = [0, 0.7071068, 0, 0.7071068]
        x_axis_negative = [0, -0.7071068, 0, 0.7071068]
        y_axis_negative = [0.7068252, 0, 0, 0.7073883]
        y_axis_positive = [-0.7068252, 0, 0, 0.7073883]
        z_axis_positive = [0, 0, 0, 1]
        z_axis_negative = [1, 0, 0, 0]
        front_side_grasp = (
            Rotation.from_quat(y_axis_positive) * Rotation.from_euler("z", -np.pi / 2)
        ).as_quat()
        self.rotation_offset_dict = {
            "top": z_axis_negative,
            "left": y_axis_positive,
            "right": y_axis_negative,
            "forward": x_axis_positive,
            "backward": x_axis_negative,
            # +Y side approach, rolled so the finger opening axis is
            # horizontal with the table rather than vertical.
            "front": front_side_grasp,
        }

        # initialize pose
        self.home_conf = [0, -0.785398163397, 0, -2.35619449, 0, 1.57079632679, 0.78539816, 0.04, 0.04]
        self.initialize_pose()

    def initialize_pose(self):
        set_joint_positions(self.robot, self.ik_joints, self.home_conf)

    def _record_motion_planner_call(self):
        self.motion_planner_call_count += 1
        increment_metric("motion_planner_calls")

    @staticmethod
    def _wrist_equivalent_orientations(theta):
        """Top-grasp orientations that close on the same faces as ``theta``.

        Only the wrist differs. Offering both signs matters because the last
        joint's range is not symmetric about every yaw: one representation can
        sit outside the limit while the other is central.
        """
        from scipy.spatial.transform import Rotation as _R

        seen = []
        for offset in (np.pi, -np.pi):
            flipped = float(theta) + offset
            if any(abs(flipped - previous) < 1e-9 for previous in seen):
                continue
            seen.append(flipped)
            yield _R.from_euler(
                "XYZ", [np.pi, 0, flipped], degrees=False).as_quat()

    def _project_small_ik_limit_spill(self, candidate, context):
        """Project only numerical IK spill onto safe planning limits."""
        projected = list(candidate)
        projected_any = False
        for index, (joint, value) in enumerate(
                zip(self.ik_joints, projected)):
            if joint not in self.planning_limits:
                continue
            lower, upper = self.planning_limits[joint]
            joint_name = p.getJointInfo(
                self.robot, joint)[1].decode("utf-8")
            if not np.isfinite(value):
                return None, (
                    f"{context} has non-finite {joint_name} position"
                )
            if lower <= value <= upper:
                continue
            safe_value = min(max(value, lower), upper)
            spill = abs(value - safe_value)
            if spill > IK_NUMERICAL_LIMIT_TOLERANCE:
                return None, (
                    f"{context} violates {joint_name} position limit: "
                    f"{value:.6f} not in [{lower:.6f}, {upper:.6f}]"
                )
            projected[index] = safe_value
            projected_any = True
        if projected_any:
            increment_metric("ik_limit_projections")
        return tuple(projected), None

    @staticmethod
    def _validated_cached_segments(traj, primitive):
        """Return a cached phase trace after checking its replay contract."""
        raw_trace = getattr(traj, "motion_trace", None)
        if not raw_trace:
            return None

        segments = []
        for item in raw_trace:
            if not isinstance(item, dict):
                raise ValueError("Cached motion trace contains a non-object segment")
            phase = item.get("phase")
            segment = item.get("trajectory")
            if not isinstance(phase, str) or not segment:
                raise ValueError("Cached motion trace has an empty or malformed segment")
            segments.append((phase, list(segment)))

        phases = tuple(phase for phase, _ in segments)
        allowed = {
            "pick": {
                ("pick_approach",),
                ("pick_approach", "post_grasp_cartesian_lift"),
                (
                    "pick_pregrasp",
                    "pick_cartesian_descent",
                    "post_grasp_cartesian_lift",
                ),
            },
            "place": {
                ("place_transfer",),
                ("place_transfer", "post_place_lift"),
                ("place_transfer", "post_place_retreat"),
            },
        }
        if phases not in allowed[primitive]:
            raise ValueError(
                f"Cached {primitive} motion trace has unsupported phases: {phases}"
            )

        for (_, previous), (_, following) in zip(segments, segments[1:]):
            if not np.allclose(
                np.asarray(previous[-1], dtype=float),
                np.asarray(following[0], dtype=float),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Cached {primitive} motion trace is discontinuous at a phase boundary"
                )

        # ``Action.traj`` is the full pick path but only the transfer portion
        # of a place path.  Check that the primary list and phase trace agree,
        # preventing a mutated path from inheriting stale phase boundaries.
        if primitive == "pick":
            traced = list(segments[0][1])
            for _, segment in segments[1:]:
                traced.extend(segment[1:])
        else:
            traced = list(segments[0][1])
        if len(traced) != len(traj) or not np.allclose(
            np.asarray(traced, dtype=float),
            np.asarray(traj, dtype=float),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"Cached {primitive} waypoints do not match the validated phase trace"
            )
        return segments

    def plan_cartesian_descent(
        self,
        start_conf,
        target_position,
        target_orientation,
        obstacles,
        step_size=0.004,
    ):
        """Plan a Cartesian segment without displaying IK sampling motion."""
        self._record_motion_planner_call()
        with LockRenderer():
            return self._plan_cartesian_descent_impl(
                start_conf=start_conf,
                target_position=target_position,
                target_orientation=target_orientation,
                obstacles=obstacles,
                step_size=step_size,
            )

    def _plan_cartesian_descent_impl(
        self,
        start_conf,
        target_position,
        target_orientation,
        obstacles,
        step_size=0.004,
    ):
        """Follow a sampled straight TCP line from ``start_conf`` to a target.

        Each Cartesian sample is solved from the preceding joint posture to
        retain the same FR3 IK branch. Every sample is checked against joint
        limits, robot self-collision, and workspace obstacles before the path
        is time-parameterized for playback.
        """
        if step_size <= 0:
            raise ValueError("Cartesian descent step_size must be positive")

        saved_conf = tuple(get_joint_positions(self.robot, self.ik_joints))
        start_conf = tuple(float(value) for value in start_conf)
        target_position = np.asarray(target_position, dtype=float)
        motion_limits = dict(self.planning_limits)
        for index, joint in enumerate(self.ik_joints):
            if joint not in self.arm_joints:
                motion_limits[joint] = (start_conf[index], start_conf[index])

        collision_fn = get_collision_fn(
            self.robot,
            self.ik_joints,
            obstacles=list(obstacles.keys()),
            attachments=self.attachments_robot,
            self_collisions=True,
            custom_limits=motion_limits,
        )

        try:
            set_joint_positions(self.robot, self.ik_joints, start_conf)
            start_position, _ = get_link_pose(
                self.robot, self.tool_attach_link)
            start_position = np.asarray(start_position, dtype=float)
            distance = float(np.linalg.norm(target_position - start_position))
            sample_count = max(2, int(np.ceil(distance / step_size)) + 1)
            path = [start_conf]
            previous_conf = start_conf

            for fraction in np.linspace(0.0, 1.0, sample_count)[1:]:
                waypoint = (
                    (1.0 - fraction) * start_position
                    + fraction * target_position
                )
                increment_metric("ik_calls")
                candidate = p.calculateInverseKinematics(
                    bodyUniqueId=self.robot,
                    endEffectorLinkIndex=self.tool_attach_link,
                    targetPosition=waypoint,
                    targetOrientation=target_orientation,
                    lowerLimits=self.ik_lower_limits,
                    upperLimits=self.ik_upper_limits,
                    jointRanges=self.ik_joint_ranges,
                    restPoses=previous_conf,
                    currentPositions=previous_conf,
                    maxNumIterations=100000,
                    residualThreshold=0.0001,
                )
                candidate = list(candidate)
                for index, joint in enumerate(self.ik_joints):
                    if joint not in self.arm_joints:
                        candidate[index] = start_conf[index]
                candidate = tuple(candidate)

                candidate, limit_feedback = (
                    self._project_small_ik_limit_spill(
                        candidate, "Cartesian descent"))
                if candidate is None:
                    return False, None, limit_feedback

                if not self.verify_ik(
                    waypoint, target_orientation, candidate):
                    return False, None, (
                        "No IK solution for Cartesian descent waypoint at "
                        f"({waypoint[0]:.3f}, {waypoint[1]:.3f}, "
                        f"{waypoint[2]:.3f})"
                    )
                increment_metric("collision_checks")
                if collision_fn(candidate):
                    return False, None, (
                        "Cartesian descent waypoint is in collision at "
                        f"({waypoint[0]:.3f}, {waypoint[1]:.3f}, "
                        f"{waypoint[2]:.3f})"
                    )

                arm_delta = max(
                    abs(candidate[index] - previous_conf[index])
                    for index, joint in enumerate(self.ik_joints)
                    if joint in self.arm_joints
                )
                if arm_delta > 0.35:
                    return False, None, (
                        "Cartesian descent IK changed configuration branch "
                        f"(joint step={arm_delta:.3f} rad)"
                    )

                path.append(candidate)
                previous_conf = candidate
                set_joint_positions(
                    self.robot, self.ik_joints, previous_conf)

            path = densify_joint_path(
                self.robot, self.ik_joints, path, max_joint_step=0.015)
            path = time_parameterize_joint_path(
                path,
                time_step=TRAJECTORY_TIME_STEP,
                max_joint_velocity=0.25,
                max_joint_acceleration=1.0,
                direction_tolerance=0.02,
            )
            return True, path, "Success"
        except ValueError as exc:
            return False, None, f"Failed Cartesian descent continuity check: {exc}"
        finally:
            set_joint_positions(self.robot, self.ik_joints, saved_conf)
            for attachment in self.attachments_robot:
                attachment.assign()

    def pick(
        self,
        object,
        obstacles,
        grasp_direction,
        traj=None,
        play_traj=True,
        grasp_yaw=0.0,
        grasp_z_offset=None,
        pregrasp_clearance=None,
        postgrasp_clearance=None,
    ):
        assert grasp_direction in self.position_offset_dict.keys(), "Unknown grasp direction!"
        self.last_motion_trace = []
        if len(self.attachments_robot) > 0:
            self.release_gripper()
        # prepare grasping pose
        current_conf = get_joint_positions(self.robot, self.ik_joints)
        box_position, box_orientation = get_pose(object)
        grasp_position_offset = np.array(
            self.position_offset_dict[grasp_direction], dtype=float)
        if grasp_direction == "top" and grasp_z_offset is not None:
            grasp_position_offset[2] = float(grasp_z_offset)
        ee_grasp_position = box_position + grasp_position_offset
        base_ori = self.rotation_offset_dict[grasp_direction]
        if grasp_yaw != 0.0:
            # Rotate the grasp orientation around world Z by grasp_yaw so the
            # gripper fingers align with the narrow axis of the object.
            ee_grasp_orientation = (
                Rotation.from_euler("z", grasp_yaw) * Rotation.from_quat(base_ori)
            ).as_quat()
        else:
            ee_grasp_orientation = base_ori

        pregrasp_traj = None
        descent_traj = None
        cached_segments = None
        if traj is None:
            planning_target = np.array(ee_grasp_position, dtype=float)
            use_vertical_pregrasp = (
                grasp_direction == "top"
                and pregrasp_clearance is not None
                and float(pregrasp_clearance) > 0.0
            )
            if use_vertical_pregrasp:
                planning_target[2] += float(pregrasp_clearance)

            success, approach_traj, feedback = self.motion_planning(
                self.robot,
                self.ik_joints,
                planning_target,
                ee_grasp_orientation,
                obstacles,
                self.attachments_robot,
                self.planning_limits,
                ik_seed_conf=current_conf,
            )
            if not success:
                success, approach_traj, fallback_feedback = self.motion_planning(
                    self.robot,
                    self.ik_joints,
                    planning_target,
                    ee_grasp_orientation,
                    obstacles,
                    self.attachments_robot,
                    self.planning_limits,
                    ik_seed_conf=self.home_conf,
                )
                if success:
                    feedback = "Success"
                    logger.debug("Pick motion repaired with home-seeded IK")
                else:
                    feedback = fallback_feedback or feedback
            if not success:
                success, approach_traj, fallback_feedback = self._plan_place_via_lifted_target(
                    planning_target,
                    ee_grasp_orientation,
                    obstacles,
                )
                if success:
                    feedback = "Success"
                    logger.debug("Pick motion repaired via lifted target waypoint")
                else:
                    feedback = fallback_feedback or feedback

            if success and use_vertical_pregrasp:
                pregrasp_traj = approach_traj
                success, descent_traj, descent_feedback = self.plan_cartesian_descent(
                    pregrasp_traj[-1],
                    ee_grasp_position,
                    ee_grasp_orientation,
                    obstacles,
                )
                if not success:
                    feedback = descent_feedback
                    traj = None
                else:
                    traj = list(pregrasp_traj)
                    if descent_traj:
                        traj.extend(list(descent_traj)[1:])
                    feedback = "Success"
            elif success:
                traj = approach_traj
        else:
            try:
                cached_segments = self._validated_cached_segments(traj, "pick")
            except ValueError as exc:
                return False, None, str(exc)
            replay_start = (
                cached_segments[0][1][0] if cached_segments else traj[0]
            )
            if not np.allclose(
                np.asarray(replay_start, dtype=float),
                np.asarray(current_conf, dtype=float),
                atol=1e-6,
                rtol=0.0,
            ):
                return False, None, (
                    "Cached trajectory start-state mismatch; discard the "
                    "cached path and invoke motion planning from the current state"
                )

            success = True
            feedback = "Success"

        # update attachments
        if success:
            ee_link_from_tcp = Pose(point=(0, 0.00, 0.0))
            cached_post_grasp_lift = None
            if cached_segments is not None:
                if cached_segments[-1][0] == "post_grasp_cartesian_lift":
                    cached_post_grasp_lift = cached_segments[-1][1]
                    segments = cached_segments[:-1]
                else:
                    segments = cached_segments
            elif pregrasp_traj is not None and descent_traj is not None:
                segments = (
                    ("pick_pregrasp", pregrasp_traj),
                    ("pick_cartesian_descent", descent_traj),
                )
            else:
                segments = (("pick_approach", traj),)
            for phase, segment in segments:
                self.last_motion_trace.append({
                    "phase": phase,
                    "trajectory": [tuple(conf) for conf in segment],
                })
                self.simulate_traj(
                    self.robot,
                    self.ik_joints,
                    self.attachments_robot,
                    None,
                    self.tool_attach_link,
                    ee_link_from_tcp,
                    segment,
                    play_traj,
                )

            # Attach at the reached grasp pose. Do not teleport the object
            # upward before attachment; the subsequent transfer begins from
            # this exact contact configuration.
            box_attach = create_attachment(self.robot, self.tool_attach_link, object)
            box_attach.assign()
            self.attachments_robot.append(box_attach)
            logger.debug(f"box_position: {box_position}")

            if cached_post_grasp_lift is not None:
                self.last_motion_trace.append({
                    "phase": "post_grasp_cartesian_lift",
                    "trajectory": [
                        tuple(conf) for conf in cached_post_grasp_lift
                    ],
                })
                self.simulate_traj(
                    self.robot,
                    self.ik_joints,
                    self.attachments_robot,
                    object,
                    self.tool_attach_link,
                    ee_link_from_tcp,
                    cached_post_grasp_lift,
                    play_traj,
                )
            elif (
                pregrasp_traj is not None
                and descent_traj is not None
            ) or (
                postgrasp_clearance is not None
                and float(postgrasp_clearance) > 0.0
            ):
                grasp_conf = tuple(get_joint_positions(
                    self.robot, self.ik_joints))
                grasp_tcp_position, grasp_tcp_orientation = get_link_pose(
                    self.robot, self.tool_attach_link)
                lift_height = (
                    float(postgrasp_clearance)
                    if postgrasp_clearance is not None else 0.06
                )
                lift_target = np.asarray(
                    grasp_tcp_position, dtype=float) + np.array(
                        [0.0, 0.0, lift_height])
                # The grasped object begins in intentional support contact.
                # Exclude only that support surface during the short vertical
                # departure; all other obstacles and self-collisions remain
                # active.
                lift_obstacles = {
                    body: name for body, name in obstacles.items()
                    if name != "table"
                }
                lift_success, lift_traj, lift_feedback = self.plan_cartesian_descent(
                    grasp_conf,
                    lift_target,
                    grasp_tcp_orientation,
                    lift_obstacles,
                )
                if not lift_success:
                    self.release_gripper()
                    return False, traj, (
                        "Attached-object vertical lift failed: "
                        f"{lift_feedback}"
                    )
                self.last_motion_trace.append({
                    "phase": "post_grasp_cartesian_lift",
                    "trajectory": [tuple(conf) for conf in lift_traj],
                })
                self.simulate_traj(
                    self.robot,
                    self.ik_joints,
                    self.attachments_robot,
                    object,
                    self.tool_attach_link,
                    ee_link_from_tcp,
                    lift_traj,
                    play_traj,
                )
                traj.extend(list(lift_traj)[1:])

            # The place transfer starts from this short vertical clearance
            # lift, without an arbitrary intermediate arm configuration.

            # set last grasp direction
            self.last_grasp_direction = grasp_direction
            self.last_grasp_position_offset = grasp_position_offset

        return success, traj, feedback

    def place(self, object, obstacles, x, y, z, theta, traj=None, play_traj=True,
              place_approach=None):
        if len(self.attachments_robot) == 0:
            return False, None, (
                "No object attached; the preceding pick did not establish the "
                "required attachment state"
            )
        self.last_motion_trace = []

        current_conf = get_joint_positions(self.robot, self.ik_joints)
        box_position, box_orientation = get_pose(object)
        logger.debug(f"box_position: {box_position}")
        logger.debug(f"box_orientation: {box_orientation}")

        new_box_position = np.array([x, y, z])

        approach = place_approach or self.last_grasp_direction
        pick_position_offset = (
            np.array(self.last_grasp_position_offset, dtype=float)
            if self.last_grasp_position_offset is not None
            else np.array(
                self.position_offset_dict[self.last_grasp_direction],
                dtype=float,
            )
        )

        if approach == "top":
            # Top-down placement: EE directly above target, theta controls yaw.
            rot = Rotation.from_euler("XYZ", [np.pi, 0, theta], degrees=False)
            ee_grasp_orientation = rot.as_quat()
            ee_grasp_position = new_box_position + (
                pick_position_offset
                if self.last_grasp_direction == "top"
                else self.position_offset_dict["top"]
            )
        elif approach == self.last_grasp_direction:
            # Place approach same as pick approach — use stored offset directly.
            ee_grasp_orientation = self.rotation_offset_dict[approach]
            ee_grasp_position = new_box_position + pick_position_offset
        else:
            # Place approach differs from pick approach (e.g. top-pick + front-place).
            # Compute EE position analytically from the stored grasp attachment transform.
            R_grasp = Rotation.from_quat(self.rotation_offset_dict[self.last_grasp_direction])
            pick_offset_world = pick_position_offset
            # Object position in EE local frame (fixed by the attachment).
            obj_in_local = R_grasp.inv().apply(-pick_offset_world)

            R_place = Rotation.from_quat(self.rotation_offset_dict[approach])
            # World-frame offset from EE to object in the new orientation.
            obj_offset_world = R_place.apply(obj_in_local)

            ee_grasp_position = new_box_position - obj_offset_world
            ee_grasp_orientation = self.rotation_offset_dict[approach]

        cached_segments = None
        if traj is None:
            success, traj, feedback = self.motion_planning(
                self.robot,
                self.ik_joints,
                ee_grasp_position,
                ee_grasp_orientation,
                obstacles,
                self.attachments_robot,
                self.planning_limits,
            )
            if not success and approach == "top":
                # A parallel jaw closes on the same two faces whether the wrist
                # is at theta or theta+pi, so the flipped wrist is the same
                # physical grasp. The last joint stops just short of a full
                # turn, so a yaw near its limit can be unreachable in one
                # representation and well inside the range in the other.
                for flipped in self._wrist_equivalent_orientations(theta):
                    retry_success, retry_traj, retry_feedback = (
                        self.motion_planning(
                            self.robot,
                            self.ik_joints,
                            ee_grasp_position,
                            flipped,
                            obstacles,
                            self.attachments_robot,
                            self.planning_limits,
                        )
                    )
                    if retry_success:
                        increment_metric("wrist_flip_recoveries")
                        success, traj, feedback = (
                            retry_success, retry_traj, retry_feedback)
                        ee_grasp_orientation = flipped
                        break
        else:
            try:
                cached_segments = self._validated_cached_segments(traj, "place")
            except ValueError as exc:
                return False, None, str(exc)
            replay_start = (
                cached_segments[0][1][0] if cached_segments else traj[0]
            )
            if not np.allclose(
                np.asarray(replay_start, dtype=float),
                np.asarray(current_conf, dtype=float),
                atol=1e-6,
                rtol=0.0,
            ):
                return False, None, (
                    "Cached trajectory start-state mismatch; discard the "
                    "cached path and invoke motion planning from the current state"
                )

            success = True
            feedback = "Success"

        # release gripper
        if success:
            transfer_traj = (
                cached_segments[0][1]
                if cached_segments is not None else traj
            )
            self.last_motion_trace.append({
                "phase": "place_transfer",
                "trajectory": [tuple(conf) for conf in transfer_traj],
            })
            ee_link_from_tcp = Pose(point=(0, 0.00, 0.05))
            # simulate action
            self.simulate_traj(
                self.robot,
                self.ik_joints,
                self.attachments_robot,
                object,
                self.tool_attach_link,
                ee_link_from_tcp,
                transfer_traj,
                play_traj,
            )

            self.release_gripper()
            logger.debug(f"Placed object {object}!")

            cached_retreat_phase = None
            cached_retreat = None
            if cached_segments is not None and len(cached_segments) == 2:
                cached_retreat_phase, cached_retreat = cached_segments[1]
            if cached_retreat is not None:
                self.last_motion_trace.append({
                    "phase": cached_retreat_phase,
                    "trajectory": [tuple(conf) for conf in cached_retreat],
                })
                self.simulate_traj(
                    self.robot,
                    self.ik_joints,
                    self.attachments_robot,
                    None,
                    self.tool_attach_link,
                    ee_link_from_tcp,
                    cached_retreat,
                    play_traj,
                )

            # Move locally away from the released object, not to a disconnected
            # home posture.  A front placement is inside a shelf bay, where a
            # vertical lift can collide with the board above; retreat through
            # the open -Y face instead.  Top placements retain a vertical lift.
            # A missing retreat phase makes the candidate rollout infeasible,
            # so try progressively shorter withdrawals (a deep bay can defeat
            # a single long query) and keep the first that plans.
            LIFT_HEIGHT = 0.10
            FRONT_RETREAT_DISTANCE = 0.12
            current_ee_pos, current_ee_ori = get_link_pose(
                self.robot, self.tool_attach_link)
            if approach == "front":
                retreat_phase = "post_place_retreat"
                retreat_targets = [
                    np.array([
                        current_ee_pos[0],
                        current_ee_pos[1] - distance,
                        current_ee_pos[2],
                    ])
                    for distance in RETREAT_DISTANCE_SCHEDULE
                ]
            else:
                retreat_phase = "post_place_lift"
                retreat_targets = [
                    np.array([
                        current_ee_pos[0],
                        current_ee_pos[1],
                        current_ee_pos[2] + height,
                    ])
                    for height in LIFT_HEIGHT_SCHEDULE
                ]
            lift_obstacles = dict(obstacles)
            lift_obstacles[object] = "just_placed_object"
            if cached_segments is None:
                lift_success = False
                lift_traj = None
                lift_feedback = "No retreat target was attempted"
                for retreat_target in retreat_targets:
                    lift_success, lift_traj, lift_feedback = self.motion_planning(
                        self.robot,
                        self.ik_joints,
                        retreat_target,
                        current_ee_ori,
                        lift_obstacles,
                        self.attachments_robot,
                        self.planning_limits,
                    )
                    if lift_success and lift_traj is not None:
                        break
                if lift_success and lift_traj is not None:
                    self.last_motion_trace.append({
                        "phase": retreat_phase,
                        "trajectory": [tuple(conf) for conf in lift_traj],
                    })
                    if play_traj:
                        for conf in lift_traj:
                            set_joint_positions(self.robot, self.ik_joints, conf)
                            replay_capture.frame()
                            wait_for_duration(TRAJECTORY_TIME_STEP)
                    else:
                        set_joint_positions(self.robot, self.ik_joints, lift_traj[-1])
                    logger.debug("Post-place local retreat successful")
                else:
                    logger.debug(
                        f"Post-place local retreat failed: {lift_feedback}")

        return success, traj, feedback

    def _plan_place_via_lifted_target(self, target_pos, target_ori, obstacles):
        """Plan place motion through lifted waypoints, avoiding home."""
        start_conf = get_joint_positions(self.robot, self.ik_joints)
        current_ee_pos, _ = get_link_pose(self.robot, self.tool_attach_link)
        high_z = min(max(current_ee_pos[2], target_pos[2]) + 0.20, 0.58)
        lifted_current = np.array([
            current_ee_pos[0],
            current_ee_pos[1],
            high_z,
        ])
        lifted_target = np.array([
            target_pos[0],
            target_pos[1],
            high_z,
        ])

        side_offsets = (0.24, -0.24, 0.34, -0.34)
        candidate_routes = [
            [lifted_target, np.array(target_pos)],
        ]
        for dx in side_offsets:
            side_x = target_pos[0] + dx
            candidate_routes.append([
                lifted_current,
                np.array([side_x, current_ee_pos[1], high_z]),
                np.array([side_x, target_pos[1], high_z]),
                lifted_target,
                np.array(target_pos),
            ])

        last_feedback = None
        for route in candidate_routes:
            success, route_traj, feedback = self._plan_ee_route(
                route, target_ori, obstacles, start_conf)
            if success:
                return True, route_traj, "Success"
            last_feedback = feedback
        return False, None, last_feedback

    def _plan_ee_route(self, route, target_ori, obstacles, start_conf):
        combined = []
        last_feedback = None

        set_joint_positions(self.robot, self.ik_joints, start_conf)
        for attachment in self.attachments_robot:
            attachment.assign()

        try:
            for waypoint in route:
                success, segment_traj, feedback = self.motion_planning(
                    self.robot,
                    self.ik_joints,
                    waypoint,
                    target_ori,
                    obstacles,
                    self.attachments_robot,
                    self.planning_limits,
                )
                if not success or segment_traj is None:
                    last_feedback = feedback
                    return False, None, last_feedback

                if combined:
                    combined.extend(list(segment_traj)[1:])
                else:
                    combined.extend(list(segment_traj))

                set_joint_positions(self.robot, self.ik_joints, segment_traj[-1])
                for attachment in self.attachments_robot:
                    attachment.assign()
            return True, combined, "Success"
        finally:
            set_joint_positions(self.robot, self.ik_joints, start_conf)
            for attachment in self.attachments_robot:
                attachment.assign()

    def release_gripper(self):
        self.attachments_robot = []
        self.last_grasp_direction = None
        self.last_grasp_position_offset = None

    def verify_ik(self, targeted_pos, targeted_ori, joint_conf):
        original_conf = get_joint_positions(self.robot, self.ik_joints)
        set_joint_positions(self.robot, self.ik_joints, joint_conf)
        pos, ori = get_link_pose(self.robot, self.tool_attach_link)

        dist = get_distance(pos, targeted_pos)
        ori_dist = quat_angle_between(ori, targeted_ori)

        set_joint_positions(self.robot, self.ik_joints, original_conf)

        if dist < 0.01 and ori_dist < 0.4:
            # print("IK solution found")
            return True
        else:
            # print("IK solution not found")
            return False

    def motion_planning(self, *args, **kwargs):
        """Plan invisibly and restore the displayed world before returning."""
        self._record_motion_planner_call()
        saved_conf = get_joint_positions(self.robot, self.ik_joints)
        with LockRenderer():
            try:
                return self._motion_planning_impl(*args, **kwargs)
            finally:
                set_joint_positions(
                    self.robot, self.ik_joints, saved_conf)
                for attachment in self.attachments_robot:
                    attachment.assign()

    def plan_joint_configuration(
        self,
        target_conf,
        obstacles,
        attachments=None,
        diagnosis=False,
    ):
        """Plan from the current state to an explicit FR3 joint posture.

        A measured pre-push posture is already a joint-space goal. This keeps
        the displayed state unchanged during planning, pins the gripper
        joints, and applies the same interpolation and dynamic limits as the
        Cartesian planner.
        """
        self._record_motion_planner_call()
        attachments = list(attachments or [])
        saved_conf = tuple(get_joint_positions(self.robot, self.ik_joints))
        requested = [float(value) for value in target_conf]
        if len(requested) == len(self.arm_joints):
            arm_values = iter(requested)
            end_conf = [
                next(arm_values) if joint in self.arm_joints else saved_conf[index]
                for index, joint in enumerate(self.ik_joints)
            ]
        elif len(requested) == len(self.ik_joints):
            end_conf = requested
            for index, joint in enumerate(self.ik_joints):
                if joint not in self.arm_joints:
                    end_conf[index] = saved_conf[index]
        else:
            return False, None, (
                "Joint goal must contain either seven FR3 arm joints or all "
                f"{len(self.ik_joints)} controllable joints; received "
                f"{len(requested)}."
            )

        for joint, value in zip(self.ik_joints, end_conf):
            if joint not in self.planning_limits:
                continue
            lower, upper = self.planning_limits[joint]
            if not lower <= value <= upper:
                joint_name = p.getJointInfo(
                    self.robot, joint)[1].decode("utf-8")
                return False, None, (
                    f"Joint goal violates {joint_name} position limit: "
                    f"{value:.6f} not in [{lower:.6f}, {upper:.6f}]"
                )

        motion_limits = dict(self.planning_limits)
        for index, joint in enumerate(self.ik_joints):
            if joint not in self.arm_joints:
                motion_limits[joint] = (saved_conf[index], saved_conf[index])

        with LockRenderer():
            try:
                path, feedback = plan_joint_motion(
                    self.robot,
                    self.ik_joints,
                    tuple(end_conf),
                    obstacles=list(obstacles.keys()),
                    attachments=attachments,
                    self_collisions=True,
                    custom_limits=motion_limits,
                    diagnosis=diagnosis,
                )
            finally:
                set_joint_positions(self.robot, self.ik_joints, saved_conf)
                for attachment in attachments:
                    attachment.assign()

        if path is None:
            return False, None, feedback

        try:
            path = densify_joint_path(
                self.robot, self.ik_joints, path, max_joint_step=0.025)
            path = time_parameterize_joint_path(
                path,
                time_step=TRAJECTORY_TIME_STEP,
                max_joint_velocity=TRAJECTORY_MAX_JOINT_VELOCITY,
                max_joint_acceleration=TRAJECTORY_MAX_JOINT_ACCELERATION,
            )
        except ValueError as exc:
            return False, None, f"Failed trajectory continuity check: {exc}"

        return True, path, "Success"

    def _motion_planning_impl(
        self,
        robot,
        ik_joints,
        robot_ee_position,
        robot_end_orientation,
        obstacles,
        attachments_robot,
        custom_limits,
        diagnosis=False,
        ik_seed_conf=None,
    ):
        original_conf = get_joint_positions(self.robot, self.ik_joints)
        motion_limits = dict(custom_limits)
        for index, joint in enumerate(self.ik_joints):
            if joint not in self.arm_joints:
                # Arm motion planning must not use gripper opening as an
                # additional random planning dimension.
                motion_limits[joint] = (
                    original_conf[index],
                    original_conf[index],
                )
        # Explicitly seed PyBullet IK with the current configuration. Without
        # currentPositions, numerically equivalent calls can converge to
        # different Franka elbow branches, producing an unnecessary full-arm
        # reconfiguration before an otherwise nearby Cartesian target.
        ik_seed = (
            tuple(ik_seed_conf)
            if ik_seed_conf is not None
            else tuple(original_conf)
        )

        # test ik first
        increment_metric("ik_calls")
        end_conf = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.tool_attach_link,
            targetPosition=robot_ee_position,
            targetOrientation=robot_end_orientation,
            lowerLimits=self.ik_lower_limits,
            upperLimits=self.ik_upper_limits,
            jointRanges=self.ik_joint_ranges,
            restPoses=ik_seed,
            currentPositions=ik_seed,
            maxNumIterations=100000,
            residualThreshold=0.0001,
        )
        # Arm IK must not silently command the independent gripper joints.
        # Keep their current values throughout pick/place arm motion.
        end_conf = list(end_conf)
        for index, joint in enumerate(self.ik_joints):
            if joint not in self.arm_joints:
                end_conf[index] = original_conf[index]
        end_conf = tuple(end_conf)

        end_conf, limit_feedback = self._project_small_ik_limit_spill(
            end_conf, "IK solution")
        if end_conf is None:
            logger.debug(limit_feedback)
            return False, None, limit_feedback

        ik_found = self.verify_ik(robot_ee_position, robot_end_orientation, end_conf)
        if not ik_found:
            logger.debug("Do not have a feasible ik")
            failure_feedback = "Failed because no IK solution exists for the grasping pose"
            return False, None, failure_feedback

        # test final ee pose if in collision
        if_collision, feedback = check_ee_collision(
            self.robot,
            self.ik_joints,
            self.tool_attach_link,
            end_conf,
            obstacles,
            self.attachments_robot,
        )
        if if_collision:
            logger.debug(feedback)
            return False, None, feedback

        # plan motion
        logger.debug(str(obstacles))
        path, feedback = plan_joint_motion(
            robot,
            ik_joints,
            end_conf,
            obstacles=list(obstacles.keys()),
            attachments=attachments_robot,
            self_collisions=True,
            custom_limits=motion_limits,
            diagnosis=diagnosis,
        )

        # process planned results
        if path is None:
            logger.debug(feedback)
            return False, None, feedback
        else:
            logger.debug("Found feasible motion plan!")

        try:
            path = densify_joint_path(
                robot, ik_joints, path, max_joint_step=0.025)
            path = time_parameterize_joint_path(
                path,
                time_step=TRAJECTORY_TIME_STEP,
                max_joint_velocity=TRAJECTORY_MAX_JOINT_VELOCITY,
                max_joint_acceleration=TRAJECTORY_MAX_JOINT_ACCELERATION,
            )
        except ValueError as exc:
            logger.warning("Rejecting invalid joint trajectory: %s", exc)
            return False, None, f"Failed trajectory continuity check: {exc}"

        for conf in path:
            for joint, value in zip(self.ik_joints, conf):
                if joint in self.planning_limits:
                    lower, upper = self.planning_limits[joint]
                    if not lower <= value <= upper:
                        joint_name = p.getJointInfo(
                            self.robot, joint
                        )[1].decode("utf-8")
                        return (
                            False,
                            None,
                            f"Trajectory violates {joint_name} position limit",
                        )

        return True, path, "Success"

    def simulate_traj(
        self,
        robot,
        ik_joints,
        attachments_robot,
        attached,
        tool_attach_link,
        ee_link_from_tcp,
        traj,
        play_traj,
    ):
        if play_traj:
            logger.debug("Simulate trajectory!")
            logger.debug(f"attachments_robot: {attachments_robot}")
            logger.debug(f"attached: {attached}")
            for conf in traj:
                set_joint_positions(robot, ik_joints, conf)
                for attachment in attachments_robot:
                    attachment.assign()
                replay_capture.frame()
                wait_for_duration(TRAJECTORY_TIME_STEP)
        else:
            # directly set end conf
            set_joint_positions(robot, ik_joints, traj[-1])
            for attachment in attachments_robot:
                attachment.assign()
