"""
Tower of Hanoi Environment for ForeSight-TAMP.

Three pegs (A=source, B=auxiliary, C=goal), N disks ordered largest→smallest.
Robot must move all disks from peg A to peg C following classic ToH rules:
  1. Only the top disk of any peg may be picked.
  2. A larger disk may never be placed on a smaller disk.
  3. All disks must end on peg C, largest at bottom.
"""
import logging
import numpy as np

import pybullet as p

from envs.pb_env import PybulletEnv
from utils.tamp_util import Action, PrimitiveAction, Parameter
from utils.llm_util import textualize_array
from utils.pb_util import create_box, set_point
from utils.dynamics_util import apply_configured_dynamics
from envs.constants import TAN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# z of table surface (boxes sit at half their height above this)
TABLE_SURFACE_Z = 0.010
DISK_H = 0.018          # height of every disk (thin so peg visually protrudes above each layer)
PEG_RADIUS = 0.13       # horizontal tolerance to assign a disk to a peg
HANOI_PREGRASP_CLEARANCE = 0.06


class TowerOfHanoiEnv(PybulletEnv):
    """
    True Tower of Hanoi: 3 pegs, N disks.

    All disks start stacked on peg A (largest at bottom).
    Goal: move them all to peg C via peg B using only valid ToH moves.
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
                    "z": Parameter("z", lower_limit=0.0, upper_limit=0.50),
                    "theta": Parameter("theta", lower_limit=-3.14, upper_limit=3.14),
                },
            ),
        }
        self._disks_ordered = []   # largest → smallest
        self._pegs = {}            # {"A": {"x":..,"y":..,"label":..}, ...}
        self._disk_heights = {}

    def __str__(self):
        return "TowerOfHanoiEnv"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, disks, pegs, use_gui=True, **kwargs):
        """
        Args:
            disks: list ordered largest→smallest, each element:
                   {"name": str, "color": [r,g,b,a], "w": float, "h": float}
            pegs:  {"A": {"x":float,"y":float,"label":str}, "B":..., "C":...}
        """
        super().reset(use_gui=use_gui)

        self._pegs = pegs
        self._disks_ordered = [d["name"] for d in disks]
        self._n_disks = len(disks)
        self._disk_widths = {d["name"]: d["w"] for d in disks}
        self._disk_heights = {d["name"]: d.get("h", DISK_H) for d in disks}

        # Table
        self.create_table()

        # Peg zone markers (non-colliding coloured squares) + visual rod
        peg_colors = {
            "A": [0.85, 0.35, 0.35, 0.55],
            "B": [0.35, 0.70, 0.35, 0.55],
            "C": [0.35, 0.45, 0.85, 0.55],
        }
        for peg_name, peg_cfg in pegs.items():
            px, py = peg_cfg["x"], peg_cfg["y"]

            # Flat zone marker — disable collisions so it never pushes disks
            marker = create_box(w=0.22, l=0.22, h=0.002,
                                color=peg_colors.get(peg_name, [0.5, 0.5, 0.5, 0.4]))
            set_point(marker, [px, py, 0.006])
            p.changeDynamics(marker, -1, mass=0, lateralFriction=0)
            p.setCollisionFilterGroupMask(marker, -1, 0, 0)  # non-collidable
            self._objects[f"peg_{peg_name}_marker"] = marker

        # All disks stacked on peg A, largest at bottom
        peg_a = pegs["A"]
        stack_z = TABLE_SURFACE_Z
        for i, disk_cfg in enumerate(disks):
            w = disk_cfg["w"]
            h = disk_cfg.get("h", DISK_H)
            yaw = disk_cfg.get("yaw", 0.0)
            z = stack_z + h / 2
            stack_z += h
            shape = disk_cfg.get("shape", "box")
            if shape == "cylinder":
                self.create_customized_cylinder(
                    name=disk_cfg["name"],
                    color=disk_cfg["color"],
                    radius=w / 2,
                    h=h,
                    x=peg_a["x"],
                    y=peg_a["y"],
                    z=z,
                )
            else:
                self.create_customized_box(
                    name=disk_cfg["name"],
                    color=disk_cfg["color"],
                    w=w,
                    l=w,
                    h=h,
                    x=peg_a["x"],
                    y=peg_a["y"],
                    z=z,
                    theta=np.pi,
                    yaw=yaw,
                )
            apply_configured_dynamics(
                self._objects[disk_cfg["name"]],
                disk_cfg,
                pybullet_module=p,
            )

        self.simulate(200)
        obs, obs_text = self.get_observation()
        return obs, obs_text

    # ------------------------------------------------------------------
    # Peg state helper
    # ------------------------------------------------------------------

    def _get_peg_state(self):
        """
        Returns dict peg_name → list[disk_name] ordered bottom→top.
        Disks not near any peg are put in a special "none" bucket.
        """
        peg_state = {peg: [] for peg in self._pegs}
        peg_state["none"] = []

        for disk_name in self._disks_ordered:
            if disk_name not in self.objects:
                continue
            pos = self.get_position(disk_name)
            assigned = False
            for peg_name, peg_cfg in self._pegs.items():
                dist = np.hypot(pos[0] - peg_cfg["x"], pos[1] - peg_cfg["y"])
                if dist < PEG_RADIUS:
                    peg_state[peg_name].append((disk_name, pos[2]))
                    assigned = True
                    break
            if not assigned:
                peg_state["none"].append((disk_name, pos[2]))

        for peg in peg_state:
            peg_state[peg].sort(key=lambda t: t[1])
            peg_state[peg] = [name for name, _ in peg_state[peg]]

        return peg_state

    def _stack_height(self, stack):
        return sum(self._disk_heights.get(d, DISK_H) for d in stack)

    def _next_place_z(self, peg_name, peg_state, disk_name=None):
        """Z for the center of the next disk placed on this peg."""
        stack = peg_state[peg_name]
        disk_h = (
            self._disk_heights.get(disk_name, DISK_H)
            if disk_name is not None
            else max(self._disk_heights.values(), default=DISK_H)
        )
        return TABLE_SURFACE_Z + self._stack_height(stack) + disk_h / 2

    def _find_disk_peg(self, disk_name, peg_state=None):
        peg_state = peg_state or self._get_peg_state()
        for peg_name, stack in peg_state.items():
            if peg_name == "none":
                continue
            if disk_name in stack:
                return peg_name
        return None

    def _rank(self, disk_name):
        if disk_name in self._disks_ordered:
            return self._disks_ordered.index(disk_name)
        return -1

    def _is_smaller_disk(self, disk_name, other_name):
        disk_rank = self._rank(disk_name)
        other_rank = self._rank(other_name)
        if disk_rank >= 0 and other_rank >= 0:
            return disk_rank > other_rank
        return self._disk_widths.get(disk_name, 0.0) < self._disk_widths.get(other_name, 0.0)

    def _peg_from_xy(self, x, y):
        for peg_name, peg_cfg in self._pegs.items():
            if np.hypot(x - peg_cfg["x"], y - peg_cfg["y"]) < PEG_RADIUS:
                return peg_name
        return None

    def describe_placement(self, x, y):
        """Name the peg a placement lands on, for symbolic candidate selection.

        Two Hanoi placements can be geometrically identical in every respect a
        feasibility map records and still differ in the only way that matters:
        which peg the disk ends up on. Selection needs that distinction stated
        symbolically, because the coordinates are deliberately withheld.
        """
        peg_name = self._peg_from_xy(x, y)
        return None if peg_name is None else f"peg {peg_name}"

    def _remove_disk_from_peg_state(self, peg_state, disk_name):
        cleaned = {peg: list(stack) for peg, stack in peg_state.items()}
        for stack in cleaned.values():
            if disk_name in stack:
                stack.remove(disk_name)
        return cleaned

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self):
        raw_obs = super().get_observation()

        # Strip internal objects from returned obs dict
        obs = {k: v for k, v in raw_obs.items()
               if not k.startswith("peg_") and k != "table"}

        peg_state = self._get_peg_state()

        # Peg descriptions
        peg_lines = []
        for peg_name in ("A", "B", "C"):
            peg_cfg = self._pegs[peg_name]
            disks_on = peg_state[peg_name]
            label = peg_cfg.get("label", peg_name)
            next_z = self._next_place_z(peg_name, peg_state)
            if disks_on:
                top = disks_on[-1]
                peg_lines.append(
                    f"  Peg {peg_name} ({label}): x={peg_cfg['x']:.3f}, "
                    f"y={peg_cfg['y']:.3f} | disks bottom→top: "
                    f"{' < '.join(disks_on)} | "
                    f"top disk: {top} | "
                    f"next place z={next_z:.4f}"
                )
            else:
                peg_lines.append(
                    f"  Peg {peg_name} ({label}): x={peg_cfg['x']:.3f}, "
                    f"y={peg_cfg['y']:.3f} | EMPTY | "
                    f"next place z={next_z:.4f}"
                )

        goal_cfg = self._pegs["C"]
        obs_text = "\n".join([
            f"GOAL: Move all {self._n_disks} disks from peg A to peg C "
            f"(x={goal_cfg['x']:.3f}, y={goal_cfg['y']:.3f}), largest at bottom.",
            f"Disk order (largest → smallest): {' > '.join(self._disks_ordered)}",
            "Current peg states:",
            *peg_lines,
            "Rules: (1) Only pick the TOP disk of a peg.  "
            "(2) Never place a LARGER disk on a SMALLER disk.  "
            "(3) Use the peg's exact (x, y) and the 'next place z' shown above.",
        ])
        return obs, obs_text

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, play_traj: bool = False):
        if action is None:
            return False, "No action is given!"
        if action.primitive not in self._primitive_actions.values():
            return False, "Unknown primitive action!"
        for obj_name in action.obj_args:
            if obj_name not in self.objects:
                return False, f"Unknown object: {obj_name}"

        traj = action.traj if action.traj is not None and len(action.traj) > 0 else None

        if action.primitive.name == "pick":
            obj_name = action.obj_args[0]
            peg_state = self._get_peg_state()
            source_peg = self._find_disk_peg(obj_name, peg_state)
            if source_peg is None:
                return False, (
                    f"Rule violation: cannot pick {obj_name}; it is not on any peg."
                )
            top_disk = peg_state[source_peg][-1] if peg_state[source_peg] else None
            if obj_name != top_disk:
                return False, (
                    f"Rule violation: cannot pick {obj_name} from peg {source_peg}; "
                    f"only the top disk may be picked and {top_disk} is on top. "
                    f"Move {top_disk} first."
                )

            obj = self.objects[obj_name]
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)
            # Exclude non-collidable peg markers from motion planning
            peg_ids = {self._objects[k] for k in self._objects if k.startswith("peg_")}
            obstacles = {bid: n for bid, n in obstacles.items() if bid not in peg_ids}

            # Exclude ALL disk objects from pick obstacles.
            # In ToH, pegs are 0.26 m apart — disks on other pegs are never in the
            # arm's path. Disks below the target on the SAME peg are also not in the
            # way (arm approaches from top). Only the table matters for collision.
            all_disk_ids = {self._objects[d] for d in self._disks_ordered
                            if d in self._objects and d != obj_name}
            obstacles = {bid: n for bid, n in obstacles.items()
                         if bid not in all_disk_ids}

            success, traj, mp_feedback = self.robot.pick(
                obj, obstacles, grasp_direction="top",
                traj=traj, play_traj=play_traj,
                # Approach from a collision-free pose, descend vertically to
                # grasp, then retract with the attached disk to this same TCP
                # height.  The following place plan therefore starts clear of
                # the source stack instead of in support contact with it.
                pregrasp_clearance=HANOI_PREGRASP_CLEARANCE,
            )
            if success:
                logger.debug(f"Picked {obj_name}")
            else:
                logger.debug(f"Pick {obj_name} failed: {mp_feedback}")

        elif action.primitive.name == "place":
            obj_name = action.obj_args[0]
            obj = self.objects[obj_name]
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)
            # Exclude non-collidable peg markers and rods from motion planning
            peg_ids = {self._objects[k] for k in self._objects if k.startswith("peg_")}
            obstacles = {bid: n for bid, n in obstacles.items() if bid not in peg_ids}

            # Compute correct z from actual peg state so the LLM only needs
            # to supply the target peg (x, y); stacking height is authoritative.
            tx, ty = action.param_args["x"], action.param_args["y"]
            peg_state = self._remove_disk_from_peg_state(
                self._get_peg_state(), obj_name)
            target_peg = self._peg_from_xy(tx, ty)
            if target_peg is not None:
                target_stack = peg_state.get(target_peg, [])
                target_top = target_stack[-1] if target_stack else None
                if target_top is not None and not self._is_smaller_disk(obj_name, target_top):
                    return False, (
                        f"Rule violation: cannot place {obj_name} on peg {target_peg}; "
                        f"{obj_name} is larger than top disk {target_top}. "
                        "Move only smaller disks onto larger disks."
                    )
                # Small buffer so the motion planner sees clearance between touching disks
                place_z = self._next_place_z(target_peg, peg_state, obj_name) + 0.003
                logger.debug(f"Place {obj_name} → peg {target_peg}, z={place_z:.4f}")
                # Exclude disks already on the target peg from obstacles:
                # they are support surfaces, not obstacles, for this placement.
                for stacked_disk in peg_state.get(target_peg, []):
                    stacked_id = self._objects.get(stacked_disk)
                    if stacked_id is not None:
                        obstacles.pop(stacked_id, None)
            else:
                return False, (
                    f"Rule violation: ({tx:.3f}, {ty:.3f}) is not a valid peg center. "
                    "Tower of Hanoi placements must use the exact x/y of peg A, B, or C."
                )

            success, traj, mp_feedback = self.robot.place(
                obj, obstacles,
                x=action.param_args["x"],
                y=action.param_args["y"],
                z=place_z,
                theta=action.param_args.get("theta", 0.0),
                traj=traj, play_traj=play_traj,
            )
            if success:
                self.simulate()
                self.theta_dict[obj_name] = action.param_args.get("theta", 0.0)
                logger.debug(f"Placed {obj_name}")
            else:
                logger.debug(f"Place {obj_name} failed: {mp_feedback}")

        if action.traj is None or len(action.traj) == 0:
            action.traj = traj

        return success, mp_feedback

    # ------------------------------------------------------------------
    # Goal check
    # ------------------------------------------------------------------

    def check_goal(self):
        peg_state = self._get_peg_state()
        disks_on_c = peg_state["C"]

        all_names = set(self._disks_ordered)
        on_c = set(disks_on_c)

        if on_c != all_names:
            missing = all_names - on_c
            elsewhere = [d for peg, lst in peg_state.items()
                         if peg != "C" for d in lst]
            parts = []
            if missing:
                parts.append(f"missing from C: {', '.join(missing)}")
            if elsewhere:
                parts.append(f"still on other pegs: {', '.join(elsewhere)}")
            return False, "; ".join(parts)

        if disks_on_c != self._disks_ordered:
            return False, (
                f"Wrong order on peg C: got {disks_on_c}, "
                f"need {self._disks_ordered} (largest first/bottom)"
            )

        return True, "Tower of Hanoi solved! All disks on peg C in correct order."

    # ------------------------------------------------------------------
    # Symbolic plan (optimal solution)
    # ------------------------------------------------------------------

    def get_symbolic_plan(self):
        """Return the optimal 2^N-1 move sequence as action strings."""
        moves = []
        self._hanoi_moves(self._n_disks, "A", "C", "B", moves)

        # Simulate peg occupancy to compute z at time of placement
        sim_state = {peg: list(self._disks_ordered) if peg == "A" else []
                     for peg in self._pegs}

        plan = []
        for from_peg, to_peg, disk in moves:
            sim_state[from_peg].remove(disk)
            place_z = self._next_place_z(to_peg, sim_state, disk)
            sim_state[to_peg].append(disk)

            px = self._pegs[to_peg]["x"]
            py = self._pegs[to_peg]["y"]
            plan.append(f"pick(['{disk}'], {{}})")
            plan.append(
                f"place(['{disk}'], "
                f"{{'x': {px:.3f}, 'y': {py:.3f}, "
                f"'z': {place_z:.4f}, 'theta': 0.0}})"
            )
        return plan

    def _hanoi_moves(self, n, src, dst, aux, moves):
        if n == 0:
            return
        self._hanoi_moves(n - 1, src, aux, dst, moves)
        moves.append((src, dst, self._disks_ordered[self._n_disks - n]))
        self._hanoi_moves(n - 1, aux, dst, src, moves)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _is_in_zone(self, obj_name, tol=0.02):
        """Check if obj is near any peg (used by ACFM code path)."""
        pos = self.get_position(obj_name)
        for peg_cfg in self._pegs.values():
            dist = np.hypot(pos[0] - peg_cfg["x"], pos[1] - peg_cfg["y"])
            if dist < PEG_RADIUS + tol:
                return True
        return False

    def create_task_instances(self, *args, **kwargs):
        pass
