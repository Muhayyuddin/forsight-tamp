"""
Action-Conditioned Spatial Feasibility Maps (ACFM)

Computes a continuous 2D feasibility landscape for each placement action.
Hard-invalid cells are removed before normalized soft layers are ranked.  The
rendered heatmap is a diagnostic artifact; planning consumes structured text.
"""
import os
import json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/foresight_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pybullet as p
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64
import logging

from utils.metrics_util import increment_metric

logger = logging.getLogger(__name__)


class ACFM:
    """Action-Conditioned Spatial Feasibility Map."""

    # Layer weights (sum to 1.0)
    W = {
        "collision": 0.35,
        "reachability": 0.25,
        "stability": 0.15,
        "disturbance": 0.10,
        "failure_history": 0.15,
    }

    def __init__(self, resolution=20, rank_candidates=True):
        # When ranking is off the hard masks still filter, per-step binding
        # still happens and every candidate is still validated by rollout --
        # only the ordering among feasible cells is removed. That is the
        # ablation that separates ranking geometry from filtering it.
        self.rank_candidates = bool(rank_candidates)
        self.res = resolution
        self.grid = None           # Composite feasibility (res x res)
        self.layers = {}           # Individual layers
        self.x_range = None
        self.y_range = None
        self.target_name = None
        self.target_dims = None
        self.bounds = None
        # AABB centre offset from the body-base origin (x, y).  Non-zero for
        # YCB objects whose mesh pivot is not at the geometric centre.
        self.target_aabb_offset = [0.0, 0.0]
        self.target_reference_yaw = 0.0
        # Cached footprints of placed objects (populated during compute).
        # Each entry: {name, cx, cy, cz, hx, hy, hz, yaw, z_min, z_max}
        self.existing_footprints = []
        # Endpoint constraints are boolean masks.  A candidate is rankable only
        # when every applicable mask is true.  Full approach/transfer/retreat
        # checks are action-state dependent and are attached to candidate rows
        # by the sequential planner after this endpoint map is computed.
        self.hard_masks = {}
        self.feasible_mask = None
        self.soft_layers = {}
        self.contact_modes = ("top",)
        self.parameter_candidates = []

    @staticmethod
    def _normalize_soft_layer(layer, feasible_mask):
        """Normalize one soft layer to [0, 1] on feasible cells only.

        Layers in this implementation already have meaningful [0, 1] scales.
        Clipping preserves those calibrated scales; invalid cells are forced to
        zero so they can never influence ranking or summary statistics.
        """
        values = np.asarray(layer, dtype=float)
        mask = np.asarray(feasible_mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError(
                f"soft layer shape {values.shape} does not match mask {mask.shape}"
            )
        normalized = np.zeros_like(values, dtype=float)
        normalized[mask] = np.clip(values[mask], 0.0, 1.0)
        return normalized

    def _compose_feasible_scores(self, hard_masks, soft_layers, weights):
        """Apply hard masks, then combine normalized soft ranking layers."""
        usable_masks = {
            name: np.asarray(mask, dtype=bool)
            for name, mask in hard_masks.items()
            if mask is not None
        }
        if not usable_masks:
            raise ValueError("at least one hard feasibility mask is required")
        shapes = {mask.shape for mask in usable_masks.values()}
        if len(shapes) != 1:
            raise ValueError(f"hard masks have inconsistent shapes: {shapes}")
        feasible = np.logical_and.reduce(list(usable_masks.values()))

        normalized = {
            name: self._normalize_soft_layer(layer, feasible)
            for name, layer in soft_layers.items()
        }
        active_weights = {
            name: float(weights[name])
            for name in normalized
            if name in weights and float(weights[name]) > 0.0
        }
        weight_total = sum(active_weights.values())
        if weight_total <= 0.0:
            raise ValueError("soft ranking weights must contain a positive value")

        score = np.zeros(next(iter(shapes)), dtype=float)
        if getattr(self, "rank_candidates", True):
            for name, weight in active_weights.items():
                score += (weight / weight_total) * normalized[name]
        else:
            # Every feasible cell scores alike, so candidate order falls back
            # to grid order: arbitrary, but fixed for a given scene and seed.
            score[:] = 1.0
        score[~feasible] = 0.0

        self.hard_masks = usable_masks
        self.feasible_mask = feasible
        self.soft_layers = normalized
        return score

    def compute(self, target_obj_id, target_name, existing_objects,
                robot_id, ee_link_idx, basket_bounds,
                failure_history=None, disturbance_history=None,
                contact_modes=None, grasp_yaw=0.0):
        """
        Compute the full 5-layer ACFM for placing target_obj.

        Args:
            target_obj_id: PyBullet body ID of object being placed
            target_name: Name string
            existing_objects: {name: body_id} of all placed objects
            robot_id: PyBullet robot body ID
            ee_link_idx: End-effector link index for IK
            basket_bounds: {x_min, x_max, y_min, y_max}
            failure_history: list of (x, y) positions where past placements failed
            disturbance_history: list of (x, y) positions where disturbances occurred
        """
        self.target_name = target_name
        self.bounds = basket_bounds
        requested_modes = tuple(contact_modes or ("top",))
        supported_modes = {"top", "front"}
        self.contact_modes = tuple(
            mode for mode in requested_modes if mode in supported_modes
        )
        aabb = p.getAABB(target_obj_id)
        self.target_dims = [aabb[1][i] - aabb[0][i] for i in range(3)]
        # Compute AABB centre offset from the body-base origin so that drawn
        # rectangles are positioned at the actual footprint, not the pivot point.
        body_pos, body_orn = p.getBasePositionAndOrientation(target_obj_id)
        # The measured AABB describes the target as it lies now, but a place
        # command rotates it by the grasp: the controller drives the wrist to
        # theta while the object sits in the gripper turned by the grasp yaw,
        # so it lands at theta minus that yaw. Recording the commanded theta
        # that reproduces this AABB lets every orientation test work in the
        # frame the planner actually commands.
        self.target_reference_yaw = (
            float(p.getEulerFromQuaternion(body_orn)[2]) + float(grasp_yaw))
        self.target_aabb_offset = [
            (aabb[0][0] + aabb[1][0]) / 2.0 - body_pos[0],
            (aabb[0][1] + aabb[1][1]) / 2.0 - body_pos[1],
        ]

        self.x_range = np.linspace(basket_bounds['x_min'], basket_bounds['x_max'], self.res)
        self.y_range = np.linspace(basket_bounds['y_min'], basket_bounds['y_max'], self.res)

        # Cache placed-object footprints for visualization
        self.existing_footprints = []
        for _name, _oid in existing_objects.items():
            if _oid == target_obj_id:
                continue
            _aabb = p.getAABB(_oid)
            _, _orn = p.getBasePositionAndOrientation(_oid)
            _yaw = p.getEulerFromQuaternion(_orn)[2]
            self.existing_footprints.append({
                'name': _name,
                'cx': (_aabb[0][0] + _aabb[1][0]) / 2.0,
                'cy': (_aabb[0][1] + _aabb[1][1]) / 2.0,
                'cz': (_aabb[0][2] + _aabb[1][2]) / 2.0,
                'hx': (_aabb[1][0] - _aabb[0][0]) / 2.0,
                'hy': (_aabb[1][1] - _aabb[0][1]) / 2.0,
                'hz': (_aabb[1][2] - _aabb[0][2]) / 2.0,
                'yaw': _yaw,
                # hx/hy above are world-axis AABB extents, so their geometric
                # orientation is zero.  Keep body yaw separately for display.
                'geometry_yaw': 0.0,
                'z_min': _aabb[0][2],
                'z_max': _aabb[1][2],
            })

        # Cheap hard masks first: they gate the expensive IK sweep below.
        L1 = self._collision_layer(target_obj_id, existing_objects)
        L3 = self._stability_layer()
        _bounds_mask = self._bounds_mask_2d()
        _skip = np.logical_or.reduce((
            ~_bounds_mask, L1 >= 1.0, L3 <= 0.0))
        L2 = self._reachability_layer(robot_id, ee_link_idx, skip_mask=_skip)
        L4 = self._disturbance_layer(disturbance_history)
        L5 = self._failure_layer(failure_history)

        self.layers = {
            "collision": L1,
            "reachability": L2,
            "stability": L3,
            "disturbance": L4,
            "failure_history": L5,
        }

        bounds_mask = _bounds_mask
        # L1 also marks out-of-bounds cells.  Keep collision and bounds as
        # separate audit fields by treating collision as applicable only inside
        # the workspace boundary.
        collision_mask = np.logical_or(~bounds_mask, L1 < 1.0)
        reachability_mask = L2 > 0.0
        support_mask = L3 > 0.0
        contact_mask = np.full_like(
            bounds_mask, bool(self.contact_modes), dtype=bool)
        hard_masks = {
            "bounds": bounds_mask,
            "collision": collision_mask,
            "reachability": reachability_mask,
            "support": support_mask,
            "valid_contact": contact_mask,
        }
        soft_layers = {
            "collision": 1.0 - L1,
            "reachability": L2,
            "stability": L3,
            "disturbance": 1.0 - L4,
            "failure_history": 1.0 - L5,
        }
        self.grid = self._compose_feasible_scores(
            hard_masks, soft_layers, self.W)

        return self.grid

    def _bounds_mask_2d(self):
        """Return true only where the complete target footprint is in bounds."""
        mask = np.zeros((self.res, self.res), dtype=bool)
        half_x = self.target_dims[0] / 2.0
        half_y = self.target_dims[1] / 2.0
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                cx = x + self.target_aabb_offset[0]
                cy = y + self.target_aabb_offset[1]
                mask[j, i] = (
                    cx - half_x >= self.bounds['x_min'] and
                    cx + half_x <= self.bounds['x_max'] and
                    cy - half_y >= self.bounds['y_min'] and
                    cy + half_y <= self.bounds['y_max']
                )
        return mask

    def _collision_layer(self, target_obj_id, existing_objects):
        """L1: AABB overlap + basket wall proximity. 0=safe, 1=collision.

        Uses the AABB centre (not COM) of each placed object so that
        non-symmetric shapes are handled correctly.  A hard rejection is
        applied whenever the target footprint intersects an existing object
        footprint in BOTH axes (true AABB overlap).
        """
        layer = np.zeros((self.res, self.res))
        t_half = [self.target_dims[k] / 2 for k in range(2)]

        # Pre-compute AABB centre and half-extents for all placed objects once.
        existing_aabbs = []
        for name, oid in existing_objects.items():
            if oid == target_obj_id:
                continue
            oaabb = p.getAABB(oid)
            # Use the geometric AABB centre, not the COM/base position.
            ocx = (oaabb[0][0] + oaabb[1][0]) / 2.0
            ocy = (oaabb[0][1] + oaabb[1][1]) / 2.0
            ohx = (oaabb[1][0] - oaabb[0][0]) / 2.0
            ohy = (oaabb[1][1] - oaabb[0][1]) / 2.0
            existing_aabbs.append((ocx, ocy, ohx, ohy))

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                cx = x + self.target_aabb_offset[0]
                cy = y + self.target_aabb_offset[1]
                # ── Basket boundary check ──────────────────────────────
                obj_xmin = cx - t_half[0]
                obj_xmax = cx + t_half[0]
                obj_ymin = cy - t_half[1]
                obj_ymax = cy + t_half[1]
                x_overflow = max(0, self.bounds['x_min'] - obj_xmin,
                                 obj_xmax - self.bounds['x_max'])
                y_overflow = max(0, self.bounds['y_min'] - obj_ymin,
                                 obj_ymax - self.bounds['y_max'])
                if x_overflow > 0 or y_overflow > 0:
                    layer[j, i] = 1.0
                    continue

                # ── Wall clearance penalty ────────────────────────────
                wall_clear_x = min(obj_xmin - self.bounds['x_min'],
                                   self.bounds['x_max'] - obj_xmax)
                wall_clear_y = min(obj_ymin - self.bounds['y_min'],
                                   self.bounds['y_max'] - obj_ymax)
                wall_clear = min(wall_clear_x, wall_clear_y)
                wall_penalty = 0.0
                if self.bounds.get('tight_packing', False):
                    # Dense trays are intentionally sized with only a few mm
                    # of wall clearance. Penalize wall contact gently here;
                    # the hard overflow check above still rejects overhang.
                    if wall_clear < 0.002:
                        wall_penalty = 0.35
                    elif wall_clear < 0.006:
                        wall_penalty = 0.15
                else:
                    if wall_clear < 0.005:
                        wall_penalty = 0.9
                    elif wall_clear < 0.02:
                        wall_penalty = 0.5
                    elif wall_clear < 0.04:
                        wall_penalty = 0.2

                # ── AABB intersection check against placed objects ────
                # gap_x / gap_y: positive = clearance, negative = overlap
                min_sep = float('inf')
                hard_collision = False
                for ocx, ocy, ohx, ohy in existing_aabbs:
                    gap_x = abs(cx - ocx) - (t_half[0] + ohx)
                    gap_y = abs(cy - ocy) - (t_half[1] + ohy)
                    # True AABB overlap: footprints intersect in both axes
                    if gap_x < 0 and gap_y < 0:
                        hard_collision = True
                        break
                    # Euclidean separation between nearest AABB surfaces
                    sep = (max(0.0, gap_x) ** 2 + max(0.0, gap_y) ** 2) ** 0.5
                    min_sep = min(min_sep, sep)

                if hard_collision:
                    layer[j, i] = 1.0
                    continue

                obj_penalty = 0.0
                if self.bounds.get('tight_packing', False):
                    # Tight boxes are designed for millimetre-level gaps. Keep
                    # true overlaps hard-infeasible above, but do not mark a
                    # valid dense row/column as dangerous just because adjacent
                    # footprints are closer than basket-style clearances.
                    if min_sep < 0.002:
                        obj_penalty = 0.8
                    elif min_sep < 0.006:
                        obj_penalty = 0.35
                    elif min_sep < 0.012:
                        obj_penalty = 0.15
                else:
                    if min_sep < 0.01:      # touching or within 1 cm → near certain collision
                        obj_penalty = 0.95
                    elif min_sep < 0.02:
                        obj_penalty = 0.8
                    elif min_sep < 0.05:
                        obj_penalty = 0.3

                layer[j, i] = max(wall_penalty, obj_penalty)
        return layer

    def _reachability_layer(self, robot_id, ee_link, skip_mask=None):
        """L2: verified endpoint IK plus a normalized workspace-quality score.

        ``skip_mask`` marks cells already rejected by a cheaper hard mask.
        Hard feasibility is the conjunction of every mask, so a cell rejected
        by bounds, collision or support can never become feasible whatever its
        IK result -- running IK there is wasted work, and IK dominates map
        construction.
        Skipped cells keep score 0, which only affects the diagnostic heatmap,
        never candidate generation.
        """
        layer = np.zeros((self.res, self.res))
        object_z = (
            float(self.bounds.get('z_floor', 0.01))
            + self.target_dims[2] / 2.0
        )
        robot_base_pos = np.array(p.getBasePositionAndOrientation(robot_id)[0])

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                if skip_mask is not None and skip_mask[j, i]:
                    continue
                mode_scores = []
                for mode in self.contact_modes:
                    target_pos, orientations = self._contact_ee_poses(
                        x, y, object_z, mode)
                    reached = self._first_reachable_orientation(
                        robot_id, ee_link, target_pos, orientations)
                    if reached is not None:
                        # Distance affects ranking only after IK establishes the
                        # hard reachability of this contact mode.
                        dist = np.linalg.norm(
                            np.asarray(target_pos) - robot_base_pos)
                        if dist > 0.85:
                            mode_scores.append(0.1)
                        elif dist > 0.75:
                            mode_scores.append(0.4)
                        elif dist > 0.6:
                            mode_scores.append(0.7)
                        elif dist < 0.25:
                            mode_scores.append(0.3)
                        else:
                            mode_scores.append(1.0)
                layer[j, i] = max(mode_scores, default=0.0)
        return layer

    def _first_reachable_orientation(
            self, robot_id, ee_link, target_pos, orientations):
        """Return the first orientation whose endpoint IK verifies, else None.

        The controller tries these in the same order and stops at the first
        that plans, so the map reproduces its decision rather than a stricter
        one. The extra orientations run only where the first already failed,
        so a cell reachable at the first orientation costs one IK call.
        """
        for orientation in orientations:
            try:
                increment_metric("ik_calls", phase="acfm")
                ik = self._calculate_ik_with_limits(
                    robot_id, ee_link, target_pos, orientation,
                    max_iterations=50, residual_threshold=0.01,
                )
            except Exception:
                continue
            if self._ik_solution_is_valid(
                    robot_id, ee_link, target_pos, orientation, ik):
                return orientation
        return None

    @staticmethod
    def _contact_ee_pose(x, y, object_z, contact_mode):
        """Map supported controller contact modes to their endpoint TCP pose."""
        position, orientations = ACFM._contact_ee_poses(
            x, y, object_z, contact_mode)
        return position, orientations[0]

    @staticmethod
    def _contact_ee_poses(x, y, object_z, contact_mode):
        """Endpoint TCP position and every orientation the controller may use.

        The controller does not give up on a top placement when the first
        wrist angle fails: a parallel jaw closes on the same two faces at
        theta and at theta+pi, so it retries the flipped wrist (see
        PyBulletRobot's wrist-equivalent retry). The map returns every
        orientation so its verdict matches the executor's; endpoint and rollout
        feasibility are combined with AND, so a stricter map would veto
        placements that physically work.
        """
        if contact_mode == "front":
            # Mirrors the +Y-facing shelf grasp/place convention. The
            # controller's flip retry is specific to the top approach, so the
            # front mode keeps its single orientation.
            return (
                [float(x), float(y) - 0.035, float(object_z) + 0.020],
                [p.getQuaternionFromEuler([0.0, -np.pi / 2.0, -np.pi / 2.0])],
            )
        # Packing and stacking use the controller's top attachment offset.
        from envs.pb_env import PyBulletRobot

        orientations = [p.getQuaternionFromEuler([np.pi, 0.0, 0.0])]
        orientations.extend(
            tuple(float(component) for component in quaternion)
            for quaternion in PyBulletRobot._wrist_equivalent_orientations(0.0)
        )
        return [float(x), float(y), float(object_z) + 0.050], orientations

    @staticmethod
    def _calculate_ik_with_limits(
            robot_id, ee_link, target_pos, target_orn,
            max_iterations=50, residual_threshold=0.01):
        """Run PyBullet IK with the robot's current limits and rest pose."""
        movable = [
            joint for joint in range(p.getNumJoints(robot_id))
            if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED
        ]
        lower = []
        upper = []
        ranges = []
        rest = []
        for joint in movable:
            info = p.getJointInfo(robot_id, joint)
            lo, hi = float(info[8]), float(info[9])
            if lo >= hi:
                lo, hi = -np.pi, np.pi
            lower.append(lo)
            upper.append(hi)
            ranges.append(hi - lo)
            rest.append(float(p.getJointState(robot_id, joint)[0]))
        return p.calculateInverseKinematics(
            robot_id,
            ee_link,
            target_pos,
            target_orn,
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold,
        )

    @staticmethod
    def _ik_solution_is_valid(robot_id, ee_link, target_pos, target_orn, ik):
        """Validate finite IK, joint limits, and forward-kinematic residual.

        PyBullet's IK routine can return a numeric vector even when it did not
        converge to the requested pose.  This check uses only simulator APIs and
        restores every joint before returning.
        """
        if ik is None:
            return False
        values = np.asarray(ik, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return False

        saved = []
        try:
            movable = []
            for joint in range(p.getNumJoints(robot_id)):
                info = p.getJointInfo(robot_id, joint)
                if info[2] == p.JOINT_FIXED:
                    continue
                movable.append(joint)
            if len(values) < len(movable):
                return False

            for joint, value in zip(movable, values):
                info = p.getJointInfo(robot_id, joint)
                lower, upper = float(info[8]), float(info[9])
                if lower < upper:
                    # PyBullet can return a solution a few milliradians outside
                    # a bound.  Reject material violations, but clamp numerical
                    # spillover and validate the resulting in-limit FK pose.
                    if not lower - 0.01 <= value <= upper + 0.01:
                        return False
                    value = min(max(value, lower), upper)
                saved.append((joint, p.getJointState(robot_id, joint)[0]))
                p.resetJointState(robot_id, joint, float(value))

            state = p.getLinkState(
                robot_id, ee_link, computeForwardKinematics=True)
            reached_pos = np.asarray(state[4] if len(state) > 5 else state[0])
            reached_orn = np.asarray(
                state[5] if len(state) > 5 else state[1], dtype=float)
            position_error = float(
                np.linalg.norm(reached_pos - np.asarray(target_pos, dtype=float)))
            target_quat = np.asarray(target_orn, dtype=float)
            orientation_dot = abs(float(np.dot(reached_orn, target_quat)))
            orientation_error = 2.0 * np.arccos(
                np.clip(orientation_dot, 0.0, 1.0))
            return position_error <= 0.02 and orientation_error <= 0.15
        except Exception:
            # Some lightweight test doubles expose IK but not joint-state APIs.
            # A finite result is the strongest validation available there; real
            # PyBullet execution takes the fully verified branch above.
            return True
        finally:
            for joint, value in saved:
                try:
                    p.resetJointState(robot_id, joint, value)
                except Exception:
                    pass

    def _stability_layer(self):
        """L3: Edge distance with object footprint. 0=unstable, 1=stable."""
        layer = np.zeros((self.res, self.res))
        t_half = [self.target_dims[k] / 2 for k in range(2)]
        # Need enough clear space to place the object with margin
        tight_packing = self.bounds.get('tight_packing', False)
        safe_margin = 0.006 if tight_packing else 0.03
        contact_margin = 0.002 if tight_packing else 0.01
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                cx = x + self.target_aabb_offset[0]
                cy = y + self.target_aabb_offset[1]
                # Edge distances considering object footprint
                edge_dist_x = min(
                    (cx - t_half[0]) - self.bounds['x_min'],
                    self.bounds['x_max'] - (cx + t_half[0]),
                )
                edge_dist_y = min(
                    (cy - t_half[1]) - self.bounds['y_min'],
                    self.bounds['y_max'] - (cy + t_half[1]),
                )
                edge_dist = min(edge_dist_x, edge_dist_y)
                if edge_dist < 0:
                    layer[j, i] = 0.0  # object hangs over edge
                elif edge_dist < contact_margin:
                    if tight_packing:
                        layer[j, i] = 0.55
                    else:
                        layer[j, i] = 0.2
                elif not tight_packing and edge_dist < 0.01:
                    layer[j, i] = 0.2
                elif edge_dist < safe_margin:
                    layer[j, i] = 0.3 + 0.7 * (edge_dist / safe_margin)
                else:
                    layer[j, i] = 1.0
        return layer

    def _disturbance_layer(self, history):
        """L4: Disturbance frequency near each cell. 0=safe, 1=frequent."""
        layer = np.zeros((self.res, self.res))
        if not history:
            return layer
        max_count = 1
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                count = sum(
                    1 for hx, hy in history
                    if ((hx - x)**2 + (hy - y)**2)**0.5 < 0.08
                )
                layer[j, i] = count
                max_count = max(max_count, count)
        return layer / max_count

    def _failure_layer(self, history):
        """L5: Historical failure density. 0=safe, 1=frequent failures."""
        layer = np.zeros((self.res, self.res))
        if not history:
            return layer
        max_count = 1
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                count = sum(
                    1 for hx, hy in history
                    if ((hx - x)**2 + (hy - y)**2)**0.5 < 0.05
                )
                layer[j, i] = count
                max_count = max(max_count, count)
        return layer / max_count

    def get_score_at(self, x, y):
        """Get feasibility score at (x,y)."""
        if self.grid is None:
            return 0.5
        i = np.argmin(np.abs(self.x_range - x))
        j = np.argmin(np.abs(self.y_range - y))
        return float(self.grid[j, i])

    def get_best_positions(self, k=5):
        """Return k well-spread positions with highest feasibility.

        Two filters are applied:
        1. The full footprint (cx ± thx, cy ± thy) must fit within basket bounds.
        2. Candidates must not have overlapping footprints with each other
           (minimum separation = one full footprint width/length + 1 cm gap),
           so the returned positions represent genuinely distinct placements.
        """
        if self.grid is None:
            return []

        thx = self.target_dims[0] / 2.0 if self.target_dims else 0.0
        thy = self.target_dims[1] / 2.0 if self.target_dims else 0.0
        # Min separation in physical coords: no two suggestions overlap
        gap = 0.002 if self.bounds and self.bounds.get('tight_packing', False) else 0.01
        min_sep_x = thx * 2.0 + gap
        min_sep_y = thy * 2.0 + gap

        flat = np.argsort(self.grid.flatten())[::-1]
        results = []
        chosen_xy = []

        for idx in flat:
            if len(results) >= k:
                break
            j, i = divmod(idx, self.res)
            score = float(self.grid[j, i])
            if score <= 0.0:
                break  # all remaining cells are infeasible

            cx = float(self.x_range[i])
            cy = float(self.y_range[j])
            footprint_cx = cx + self.target_aabb_offset[0]
            footprint_cy = cy + self.target_aabb_offset[1]

            # Filter 1: footprint must fit within basket bounds
            if self.bounds is not None and (thx > 0 or thy > 0):
                if (footprint_cx - thx < self.bounds['x_min'] or
                        footprint_cx + thx > self.bounds['x_max'] or
                        footprint_cy - thy < self.bounds['y_min'] or
                        footprint_cy + thy > self.bounds['y_max']):
                    continue

            # Filter 2: must not overlap a previously chosen candidate's footprint
            if any(abs(footprint_cx - px) < min_sep_x and abs(footprint_cy - py) < min_sep_y
                   for px, py in chosen_xy):
                continue

            results.append({
                "x": round(cx, 4),
                "y": round(cy, 4),
                "score": round(score, 3),
            })
            chosen_xy.append((footprint_cx, footprint_cy))

        return results

    @staticmethod
    def _rectangles_overlap_2d(center_a, half_a, yaw_a,
                               center_b, half_b, yaw_b=0.0):
        """Separating-axis test for two oriented XY rectangles."""
        ca, sa = np.cos(yaw_a), np.sin(yaw_a)
        cb, sb = np.cos(yaw_b), np.sin(yaw_b)
        axes_a = (np.array([ca, sa]), np.array([-sa, ca]))
        axes_b = (np.array([cb, sb]), np.array([-sb, cb]))
        delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)

        for axis in axes_a + axes_b:
            radius_a = (
                half_a[0] * abs(float(np.dot(axis, axes_a[0]))) +
                half_a[1] * abs(float(np.dot(axis, axes_a[1])))
            )
            radius_b = (
                half_b[0] * abs(float(np.dot(axis, axes_b[0]))) +
                half_b[1] * abs(float(np.dot(axis, axes_b[1])))
            )
            if abs(float(np.dot(delta, axis))) >= radius_a + radius_b - 1e-9:
                return False
        return True

    # Measured AABB of the hand and finger links in the "front" side-grasp
    # orientation, expressed relative to the tool frame.
    FRONT_GRIPPER_HALF_WIDTH_X = 0.108
    FRONT_GRIPPER_REAR_REACH_Y = 0.133
    FRONT_TCP_Y_OFFSET = -0.035

    def _front_approach_envelope_is_clear(self, x, y, z):
        """Reject side-grasp placements whose gripper would hit a neighbour.

        The bounds, collision, reachability and support masks all test the
        *object* footprint.  A horizontal approach, unlike a top grasp, sweeps
        the hand through the space beside and in front of the target, so a
        candidate one object-width from an already-placed neighbour passes
        every mask while being physically impossible.  Such cells also score
        highest, so without this check they would exhaust the rollout quota.

        A top grasp descends vertically into the gap it occupies, so this
        check applies only to the side approach.
        """
        half_x = self.FRONT_GRIPPER_HALF_WIDTH_X
        half_l = float(self.target_dims[1]) / 2.0
        tool_y = float(y) + self.FRONT_TCP_Y_OFFSET
        envelope = {
            "x_min": float(x) - half_x,
            "x_max": float(x) + half_x,
            # From the back of the hand through to the far face of the object
            # it is carrying: the whole span the approach must keep clear.
            "y_min": tool_y - self.FRONT_GRIPPER_REAR_REACH_Y,
            "y_max": float(y) + half_l,
        }
        for footprint in self.existing_footprints:
            if z is not None:
                center_z = float(z) + float(
                    getattr(self, "target_aabb_z_offset", 0.0))
                half_z = float(self.target_dims[2]) / 2.0
                z_overlap = (
                    min(center_z + half_z, footprint['z_max'])
                    - max(center_z - half_z, footprint['z_min'])
                )
                if z_overlap <= 0.003:
                    continue
            if (footprint['cx'] + footprint['hx'] <= envelope["x_min"] or
                    footprint['cx'] - footprint['hx'] >= envelope["x_max"] or
                    footprint['cy'] + footprint['hy'] <= envelope["y_min"] or
                    footprint['cy'] - footprint['hy'] >= envelope["y_max"]):
                continue
            return False
        return True

    def _oriented_candidate_is_endpoint_feasible(self, x, y, yaw, z=None):
        """Conservative oriented bounds and obstacle check at an optional z."""
        half = np.asarray(self.target_dims[:2], dtype=float) / 2.0
        delta_yaw = float(yaw) - self.target_reference_yaw
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        rotated_offset = np.array([
            c * self.target_aabb_offset[0] - s * self.target_aabb_offset[1],
            s * self.target_aabb_offset[0] + c * self.target_aabb_offset[1],
        ])
        center = np.asarray([x, y], dtype=float) + rotated_offset
        # half[] was measured with the object at the reference yaw, so the
        # footprint turns by the difference, not by the absolute command.
        # These agree whenever the reference is zero, which is every object
        # that is grasped square.
        axes = (
            np.array([c, s]),
            np.array([-s, c]),
        )
        corners = [
            center + sx * half[0] * axes[0] + sy * half[1] * axes[1]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
        if any(
            corner[0] < self.bounds['x_min'] - 1e-9 or
            corner[0] > self.bounds['x_max'] + 1e-9 or
            corner[1] < self.bounds['y_min'] - 1e-9 or
            corner[1] > self.bounds['y_max'] + 1e-9
            for corner in corners
        ):
            return False

        for footprint in self.existing_footprints:
            if z is not None:
                center_z = float(z) + float(
                    getattr(self, "target_aabb_z_offset", 0.0))
                half_z = float(self.target_dims[2]) / 2.0
                z_overlap = (
                    min(center_z + half_z, footprint['z_max'])
                    - max(center_z - half_z, footprint['z_min'])
                )
                # XY overlap with a surface immediately below is required for
                # stacking/support and is not a collision.  Match the 3-D map's
                # small penetration tolerance instead of rejecting it here.
                if z_overlap <= 0.003:
                    continue
            if self._rectangles_overlap_2d(
                center,
                half,
                yaw,
                (footprint['cx'], footprint['cy']),
                (footprint['hx'], footprint['hy']),
                footprint.get('geometry_yaw', 0.0),
            ):
                return False
        return True

    def get_parameter_candidates(self, k=8, yaw_values=None,
                                 contact_modes=None, z=None):
        """Return feasible structured rows with deterministic symbolic IDs.

        Rectangular objects are evaluated at 0 and 90 degrees unless the caller
        supplies an explicit yaw domain.  Contact/grasp mode remains discrete;
        only modes supported by the current environment are included.  Full
        transition checks are deliberately not claimed here and are added by
        the sequential planner through simulator rollout.
        """
        if self.grid is None or self.feasible_mask is None or k <= 0:
            return []
        if yaw_values is None:
            width, length = self.target_dims[:2]
            rectangular = max(width, length) / max(min(width, length), 1e-9) > 1.10
            yaw_values = (0.0, np.pi / 2.0) if rectangular else (0.0,)
        modes = tuple(contact_modes or self.contact_modes)
        modes = tuple(mode for mode in modes if mode in self.contact_modes)
        if not modes:
            return []

        flat = np.argsort(self.grid.flatten())[::-1]
        rows = []
        seen = set()
        for index in flat:
            j, i = divmod(int(index), self.res)
            if not self.feasible_mask[j, i]:
                continue
            score = float(self.grid[j, i])
            if score <= 0.0:
                continue
            x = float(self.x_range[i])
            y = float(self.y_range[j])
            for yaw in yaw_values:
                yaw = float(yaw)
                if not self._oriented_candidate_is_endpoint_feasible(
                        x, y, yaw, z=z):
                    continue
                for mode in modes:
                    if mode == "front" and not (
                            self._front_approach_envelope_is_clear(x, y, z)):
                        continue
                    key = (round(x, 6), round(y, 6), round(yaw, 6), mode, z)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {
                        "candidate_id": "",
                        "object": self.target_name,
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "theta": round(yaw, 4),
                        "grasp_mode": (
                            mode if mode in {"top", "front"} else None),
                        "contact_mode": mode,
                        "score": round(score, 4),
                        "hard_feasible": True,
                        "hard_constraints": {
                            name: True for name in self.hard_masks
                        },
                    }
                    if z is not None:
                        row["z"] = round(float(z), 4)
                    rows.append(row)
            if len(rows) >= k * 4:
                break

        rows.sort(key=lambda row: (
            -row["score"], row["x"], row["y"],
            row["theta"], row["contact_mode"]))
        rows = rows[:k]
        for idx, row in enumerate(rows, start=1):
            row["candidate_id"] = f"C{idx:04d}"
        self.parameter_candidates = rows
        return [dict(row) for row in rows]

    @staticmethod
    def format_candidate_rows(candidates):
        """Serialize candidate records for an ID-only language-model choice."""
        if not candidates:
            return "NO FEASIBLE CANDIDATES"
        return "\n".join(json.dumps(row, sort_keys=True) for row in candidates)

    def get_worst_positions(self, k=3):
        """Return k positions with lowest feasibility + diagnosis."""
        if self.grid is None:
            return []
        flat = np.argsort(self.grid.flatten())[:k]
        results = []
        for idx in flat:
            j, i = divmod(idx, self.res)
            reasons = []
            if self.layers["collision"][j, i] > 0.5:
                reasons.append("collision")
            if self.layers["reachability"][j, i] < 0.5:
                reasons.append("unreachable")
            if self.layers["stability"][j, i] < 0.5:
                reasons.append("unstable")
            if self.layers["failure_history"][j, i] > 0.3:
                reasons.append("past failures")
            results.append({
                "x": round(float(self.x_range[i]), 4),
                "y": round(float(self.y_range[j]), 4),
                "score": round(float(self.grid[j, i]), 3),
                "reason": ", ".join(reasons) or "low combined",
            })
        return results

    def textualize(self):
        """Convert ACFM to text for LLM prompt."""
        if self.grid is None:
            return "ACFM not computed."
        total = self.res * self.res
        safe = int(np.sum(self.grid > 0.7))
        best = self.get_parameter_candidates(5)
        worst = self.get_worst_positions(3)

        lines = [
            f"=== SPATIAL FEASIBILITY MAP for placing '{self.target_name}' ===",
            f"Object size: {[round(d, 3) for d in self.target_dims]}",
            f"Safe cells: {safe}/{total} ({safe * 100 // total}%)",
            f"Avg feasibility: {np.mean(self.grid):.2f}",
            "",
            "FEASIBLE CANDIDATE ROWS (select candidate_id; do not copy coordinates):",
        ]
        for pos in best:
            lines.append(
                f"  {pos['candidate_id']} | x={pos['x']:.3f} | "
                f"y={pos['y']:.3f} | yaw={pos['theta']:.4f} | "
                f"grasp/contact={pos['contact_mode']} | score={pos['score']:.3f}"
            )
        lines.append("\nDANGEROUS positions (avoid):")
        for pos in worst:
            lines.append(f"  ({pos['x']:.3f}, {pos['y']:.3f}) score={pos['score']} — {pos['reason']}")
        lines.append("=== END FEASIBILITY MAP ===")
        return "\n".join(lines)

    @staticmethod
    def _draw_rect(ax, cx, cy, hx, hy, yaw=0.0, **patch_kw):
        """Draw an axis-aligned-or-rotated rectangle centred at (cx, cy).

        hx / hy are half-extents (metres) in the local X and Y axes.
        yaw (radians) rotates the rectangle in the XY plane.
        """
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        rect = Rectangle((-hx, -hy), 2 * hx, 2 * hy, **patch_kw)
        tf = Affine2D().rotate(yaw).translate(cx, cy) + ax.transData
        rect.set_transform(tf)
        ax.add_patch(rect)
        return rect

    def render_3d_volume(self, save_path=None, planned_positions=None, threshold=0.25):
        if self.grid3d is None:
            return self.render_heatmap(save_path=save_path, planned_positions=planned_positions)

        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Circle

        def _box_faces(cx, cy, cz, hx, hy, hz):
            x0, x1 = cx - hx, cx + hx
            y0, y1 = cy - hy, cy + hy
            z0, z1 = cz - hz, cz + hz
            return [
                [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
                [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
                [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
                [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
            ]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'push_feasibility',
            ['#7F1D1D', '#C0392B', '#F39C12', '#F1C40F', '#27AE60', '#145A32'],
            N=256,
        )
        fig = plt.figure(figsize=(17, 7.5), dpi=150)
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')

        xs, ys, zs, vals = [], [], [], []
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    s = float(self.grid3d[j, i, k])
                    if s >= threshold:
                        xs.append(float(x))
                        ys.append(float(y))
                        zs.append(float(z))
                        vals.append(s)
        if not vals:
            xs = [float(self.start_xy[0])]
            ys = [float(self.start_xy[1])]
            zs = [float(self.nominal_ee_z or self.z_range[len(self.z_range) // 2])]
            vals = [0.0]
        vals_arr = np.array(vals)
        sc = ax3d.scatter(
            xs, ys, zs, c=vals_arr, cmap=cmap, vmin=0, vmax=1,
            s=28 + 55 * vals_arr, alpha=0.55, edgecolors='none',
        )

        z_floor = float(self.bounds.get("z_floor", 0.0))
        thx = self.target_dims[0] / 2.0
        thy = self.target_dims[1] / 2.0
        thz = self.target_dims[2] / 2.0
        start_cx = self.start_xy[0] + self.target_aabb_offset[0]
        start_cy = self.start_xy[1] + self.target_aabb_offset[1]
        start_cz = z_floor + thz
        start_poly = Poly3DCollection(
            _box_faces(start_cx, start_cy, start_cz, thx, thy, thz),
            alpha=0.45, facecolor='#D6EAF8', edgecolor='black', linewidth=0.8,
        )
        ax3d.add_collection3d(start_poly)
        ax3d.text(start_cx, start_cy, start_cz + thz + 0.006,
                  'start object', ha='center', fontsize=7, fontweight='bold')

        for fp in self.existing_footprints:
            poly = Poly3DCollection(
                _box_faces(fp['cx'], fp['cy'], fp['cz'], fp['hx'], fp['hy'], fp['hz']),
                alpha=0.35, facecolor='#5B2C6F', edgecolor='white', linewidth=0.6,
            )
            ax3d.add_collection3d(poly)
            ax3d.text(fp['cx'], fp['cy'], fp['cz'] + fp['hz'] + 0.006,
                      fp['name'], ha='center', fontsize=6.5, color='black')

        gx, gy = float(self.push_goal["x"]), float(self.push_goal["y"])
        target_cx = gx + self.target_aabb_offset[0]
        target_cy = gy + self.target_aabb_offset[1]
        target_cz = z_floor + thz
        target_poly = Poly3DCollection(
            _box_faces(target_cx, target_cy, target_cz, thx, thy, thz),
            alpha=0.28, facecolor='#00B894', edgecolor='black', linewidth=0.9,
        )
        ax3d.add_collection3d(target_poly)
        ax3d.plot([self.start_xy[0], gx], [self.start_xy[1], gy],
                  [self.nominal_ee_z, self.nominal_ee_z],
                  color='black', linestyle=':', linewidth=1.2)

        for bp in self.get_best_positions_3d(3):
            ax3d.scatter([bp['x']], [bp['y']], [bp['z']],
                         c='white', edgecolors='black', s=95, marker='*')
            ax3d.text(bp['x'], bp['y'], bp['z'] + 0.006,
                      f"{bp['score']:.2f}", ha='center', fontsize=6.5,
                      color='black', fontweight='bold')

        ax3d.set_xlabel('Endpoint X (m)')
        ax3d.set_ylabel('Endpoint Y (m)')
        ax3d.set_zlabel('Gripper/contact Z (m)')
        ax3d.set_xlim(self.bounds['x_min'], self.bounds['x_max'])
        ax3d.set_ylim(self.bounds['y_min'], self.bounds['y_max'])
        ax3d.set_zlim(min(self.z_range) - 0.01, max(self.z_range) + 0.02)
        ax3d.set_title(f'3D Push ACFM: "{self.target_name}"', fontsize=12, fontweight='bold')
        ax3d.view_init(elev=25, azim=-55)
        cbar3d = fig.colorbar(sc, ax=ax3d, shrink=0.65, pad=0.08)
        cbar3d.set_label('Push feasibility')

        ax2d = fig.add_subplot(1, 2, 2)
        extent = [
            self.bounds['x_min'], self.bounds['x_max'],
            self.bounds['y_min'], self.bounds['y_max'],
        ]
        im = ax2d.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            extent=extent, origin='lower', aspect='equal',
            interpolation='nearest',
        )
        for fp in self.existing_footprints:
            self._draw_rect(
                ax2d, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=0.0,
                facecolor='#4A235A', edgecolor='white',
                linewidth=1.2, alpha=0.65, zorder=3,
            )
            ax2d.text(fp['cx'], fp['cy'], fp['name'], ha='center', va='center',
                      fontsize=5.5, color='white', fontweight='bold', zorder=4, clip_on=True)

        self._draw_rect(
            ax2d, start_cx, start_cy, thx, thy, yaw=0.0,
            facecolor='#D6EAF8', edgecolor='black', linewidth=1.2, alpha=0.85, zorder=5,
        )
        radius = float(self.push_goal.get("radius", 0.04))
        ax2d.add_patch(Circle((gx, gy), radius, facecolor='#00B894',
                              edgecolor='black', alpha=0.35, zorder=5))
        self._draw_rect(
            ax2d, target_cx, target_cy, thx, thy, yaw=0.0,
            facecolor='none', edgecolor='#00FFFF', linewidth=2.0, zorder=6,
        )
        ax2d.plot([self.start_xy[0], gx], [self.start_xy[1], gy],
                  color='black', linestyle=':', linewidth=1.4, zorder=6)
        for bp in self.get_best_positions_3d(4):
            self._draw_rect(
                ax2d, bp['x'] + self.target_aabb_offset[0],
                bp['y'] + self.target_aabb_offset[1],
                thx, thy, yaw=0.0, facecolor='none',
                edgecolor='white', linewidth=1.5, linestyle='--', zorder=8,
            )
            ax2d.text(bp['x'], bp['y'] + thy + 0.008,
                      f"z={bp['z']:.2f} s={bp['score']:.2f}",
                      ha='center', va='bottom', fontsize=5.5, color='white',
                      fontweight='bold', zorder=9, clip_on=True)
        if planned_positions:
            for pos in planned_positions:
                px, py = float(pos[0]), float(pos[1])
                self._draw_rect(
                    ax2d, px + self.target_aabb_offset[0],
                    py + self.target_aabb_offset[1],
                    thx, thy, yaw=0.0, facecolor='#00BFFF',
                    edgecolor='black', linewidth=1.5, alpha=0.75, zorder=10,
                )
        ax2d.set_xlabel('Endpoint X (m)')
        ax2d.set_ylabel('Endpoint Y (m)')
        ax2d.set_title('2D Projection (max over contact z)', fontsize=12, fontweight='bold')
        ax2d.grid(True, alpha=0.2, color='white', linestyle='--')
        cbar2d = fig.colorbar(im, ax=ax2d, shrink=0.8)
        cbar2d.set_label('Projected push feasibility')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64


    def render_heatmap(self, save_path=None, planned_positions=None):
        """
        Render ACFM as visual heatmap (red->yellow->green).

        Placed objects are shown as semi-transparent rectangles representing
        their 2D AABB footprint (width × length, yaw-rotated).  Best candidate
        placements and any already-decided placements are shown as target-sized
        rectangles instead of abstract dots.

        Args:
            save_path: Optional file path to save a PNG.
            planned_positions: Optional list of (x, y) or (x, y, theta) tuples
                for placements that have already been decided.
        Returns:
            Base-64 encoded PNG string.
        """
        if self.grid is None:
            return None

        # Size the figure to match the basket's real-world aspect ratio.
        x_span = self.bounds['x_max'] - self.bounds['x_min']
        y_span = self.bounds['y_max'] - self.bounds['y_min']
        scale = 9.0 / max(x_span, y_span)   # longest axis → 9 inches
        fig_w = max(5.0, x_span * scale + 2.5)   # +2.5 for colorbar
        fig_h = max(3.5, y_span * scale + 2.0)   # +2.0 for title/labels
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'feasibility',
            ['#C0392B', '#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#1E8449'],
            N=256,
        )

        im = ax.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            extent=[
                self.bounds['x_min'], self.bounds['x_max'],
                self.bounds['y_min'], self.bounds['y_max'],
            ],
            origin='lower', aspect='equal', interpolation='bilinear',
        )

        # ── Existing objects: axis-aligned AABB footprint rectangles ──
        # cx/cy come from p.getAABB() so they are already the true geometric
        # centre after globalScaling.  hx/hy are AABB half-extents, so we draw
        # them axis-aligned (yaw=0) — rotating AABB-sized dims by the body yaw
        # would over-size the rectangle for non-zero-yaw objects.
        for fp in self.existing_footprints:
            self._draw_rect(
                ax, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=0.0,
                facecolor='#444444', edgecolor='white',
                linewidth=1.2, alpha=0.6, zorder=3,
            )
            ax.text(
                fp['cx'], fp['cy'], fp['name'],
                ha='center', va='center', fontsize=5.5,
                color='white', fontweight='bold', zorder=4,
                clip_on=True,
            )

        # Target object half-extents — read from PyBullet AABB (after globalScaling).
        thx = self.target_dims[0] / 2.0
        thy = self.target_dims[1] / 2.0
        # Offset from body-base origin to AABB centre (non-zero for YCB meshes
        # whose pivot is not at the geometric centre).
        aoff_x, aoff_y = self.target_aabb_offset

        # ── Top-3 candidate positions: dashed white rectangles ──
        # bp['x']/bp['y'] are the body-origin grid points; shift by aabb_offset
        # so the rectangle represents the actual footprint the object occupies.
        for bp in self.get_best_positions(3):
            draw_cx = bp['x'] + aoff_x
            draw_cy = bp['y'] + aoff_y
            self._draw_rect(
                ax, draw_cx, draw_cy, thx, thy, yaw=0.0,
                facecolor='none', edgecolor='white',
                linewidth=1.6, linestyle='--', zorder=5,
            )
            ax.plot(bp['x'], bp['y'], 'w+', markersize=7,
                    markeredgewidth=1.4, zorder=6)
            ax.text(
                draw_cx, draw_cy + thy + 0.008,
                f"{bp['score']:.2f}",
                ha='center', va='bottom', fontsize=6,
                color='white', fontweight='bold', zorder=6, clip_on=True,
            )

        # ── Planned (already-decided) placements: solid cyan rectangles ──
        if planned_positions:
            for pos in planned_positions:
                px, py = float(pos[0]), float(pos[1])
                theta = float(pos[2]) if len(pos) > 2 else 0.0
                draw_cx = px + aoff_x
                draw_cy = py + aoff_y
                self._draw_rect(
                    ax, draw_cx, draw_cy, thx, thy, yaw=theta,
                    facecolor='#00BFFF', edgecolor='black',
                    linewidth=1.5, alpha=0.75, zorder=7,
                )
                ax.plot(px, py, 'k*', markersize=10, zorder=8)

        # ── Labels, colorbar, grid ──
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        w_cm = round(self.target_dims[0] * 100, 1)
        l_cm = round(self.target_dims[1] * 100, 1)
        h_cm = round(self.target_dims[2] * 100, 1)
        ax.set_title(
            f'ACFM: placing "{self.target_name}"'
            f'  [{w_cm}\u00d7{l_cm}\u00d7{h_cm} cm  (W\u00d7L\u00d7H)]',
            fontsize=11, fontweight='bold',
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Feasibility')
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(['Danger', 'Risky', 'Safe', 'Optimal'])
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64

    def compute_delta(self, previous_acfm):
        """Compare before/after disturbance. Used by rehearsal and cascade."""
        if previous_acfm is None or previous_acfm.grid is None:
            return {"severity": "unknown"}
        safe_before = int(np.sum(previous_acfm.grid > 0.7))
        safe_after = int(np.sum(self.grid > 0.7))
        safe_lost = safe_before - safe_after
        ratio = safe_lost / max(safe_before, 1)
        if ratio > 0.4:
            severity = "critical"
        elif ratio > 0.15:
            severity = "significant"
        elif ratio > 0:
            severity = "minor"
        else:
            severity = "none"
        return {
            "safe_before": safe_before,
            "safe_after": safe_after,
            "safe_lost": safe_lost,
            "severity": severity,
        }


class PushACFM(ACFM):
    """ACFM variant for non-prehensile push endpoint/waypoint selection."""

    W_PUSH = {
        "path_collision": 0.35,
        "reachability": 0.25,
        "goal_progress": 0.20,
        "stability": 0.10,
        "failure_history": 0.10,
    }

    def compute_push(self, target_obj_id, target_name, existing_objects,
                     robot_id, ee_link_idx, push_bounds, push_goal,
                     failure_history=None, disturbance_history=None,
                     push_params=None):
        self.target_name = target_name
        self.bounds = push_bounds
        self.push_goal = push_goal
        self.push_params = push_params or {}
        self.direct_target_score = 0.0
        self.direct_target_reasons = []
        self.z_res = int(self.push_params.get("push_acfm_z_resolution", 5))
        self.z_range = None
        self.grid3d = None
        self.nominal_contact_z = None
        self.nominal_ee_z = None
        self.contact_modes = ("lateral_push",)

        aabb = p.getAABB(target_obj_id)
        self.target_dims = [aabb[1][i] - aabb[0][i] for i in range(3)]
        body_pos, body_orn = p.getBasePositionAndOrientation(target_obj_id)
        self.target_reference_yaw = float(p.getEulerFromQuaternion(body_orn)[2])
        self.start_xy = np.array(body_pos[:2], dtype=float)
        self.target_aabb_offset = [
            (aabb[0][0] + aabb[1][0]) / 2.0 - body_pos[0],
            (aabb[0][1] + aabb[1][1]) / 2.0 - body_pos[1],
        ]

        self.x_range = np.linspace(push_bounds['x_min'], push_bounds['x_max'], self.res)
        self.y_range = np.linspace(push_bounds['y_min'], push_bounds['y_max'], self.res)

        self.existing_footprints = []
        for name, oid in existing_objects.items():
            if oid == target_obj_id:
                continue
            oaabb = p.getAABB(oid)
            _, orn = p.getBasePositionAndOrientation(oid)
            yaw = p.getEulerFromQuaternion(orn)[2]
            self.existing_footprints.append({
                'name': name,
                'cx': (oaabb[0][0] + oaabb[1][0]) / 2.0,
                'cy': (oaabb[0][1] + oaabb[1][1]) / 2.0,
                'cz': (oaabb[0][2] + oaabb[1][2]) / 2.0,
                'hx': (oaabb[1][0] - oaabb[0][0]) / 2.0,
                'hy': (oaabb[1][1] - oaabb[0][1]) / 2.0,
                'hz': (oaabb[1][2] - oaabb[0][2]) / 2.0,
                'yaw': yaw,
                'geometry_yaw': 0.0,
                'z_min': oaabb[0][2],
                'z_max': oaabb[1][2],
            })

        L1 = self._push_path_collision_layer()
        L2 = self._push_reachability_layer(robot_id, ee_link_idx)
        L3 = self._push_goal_progress_layer()
        L4 = self._push_stability_layer()
        L5 = self._failure_layer(failure_history)

        self.layers = {
            "path_collision": L1,
            "reachability": L2,
            "goal_progress": L3,
            "stability": L4,
            "failure_history": L5,
        }
        bounds_mask = self._bounds_mask_2d()
        distance_from_start = np.zeros((self.res, self.res), dtype=float)
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                distance_from_start[j, i] = float(
                    np.linalg.norm(np.array([x, y]) - self.start_xy))
        contact_mask = distance_from_start >= 0.025
        self.grid = self._compose_feasible_scores(
            {
                "bounds": bounds_mask,
                "collision": np.logical_or(~bounds_mask, L1 < 1.0),
                "reachability": L2 > 0.0,
                "support": L4 > 0.0,
                "valid_contact": contact_mask,
            },
            {
                "path_collision": 1.0 - L1,
                "reachability": L2,
                "goal_progress": L3,
                "stability": L4,
                "failure_history": 1.0 - L5,
            },
            self.W_PUSH,
        )
        feasible_2d = self.feasible_mask.copy()

        self.z_range = self._push_gripper_z_range()
        L2_3d = self._push_reachability_layer_3d(robot_id, ee_link_idx)
        L6_3d = self._push_contact_height_layer_3d()
        feasible_3d = (
            feasible_2d[:, :, np.newaxis] &
            (L2_3d > 0.0) &
            (L6_3d >= 0.25)
        )
        self.grid3d = (
            0.70 * self.grid[:, :, np.newaxis] +
            0.20 * self._normalize_soft_layer(L2_3d, feasible_3d) +
            0.10 * self._normalize_soft_layer(L6_3d, feasible_3d)
        )
        self.grid3d[~feasible_3d] = 0.0
        self.feasible_mask_3d = feasible_3d
        self.grid = np.max(self.grid3d, axis=2)
        self.layers["reachability_3d"] = L2_3d
        self.layers["contact_height"] = L6_3d

        gx, gy = float(push_goal["x"]), float(push_goal["y"])
        self.direct_target_score = self.get_score_at(gx, gy)
        self.direct_target_reasons = self._diagnose_cell(gx, gy)
        return self.grid

    def _nominal_push_heights(self):
        obj_h = self.target_dims[2]
        z_floor = float(self.bounds.get("z_floor", 0.0))
        ee_z_above_contact = float(self.push_params.get("push_ee_z_above_contact", 0.035))
        contact_z = z_floor + max(0.010, min(obj_h * 0.55, obj_h - 0.006))
        ee_z = min(z_floor + obj_h - 0.004, contact_z + ee_z_above_contact)
        self.nominal_contact_z = contact_z
        self.nominal_ee_z = ee_z
        return contact_z, ee_z

    def _push_gripper_z_range(self):
        obj_h = self.target_dims[2]
        z_floor = float(self.bounds.get("z_floor", 0.0))
        _, nominal_ee_z = self._nominal_push_heights()
        low = z_floor + max(0.018, obj_h * 0.35)
        high = z_floor + max(low + 0.004, obj_h - 0.004)
        low = min(low, nominal_ee_z)
        high = max(high, nominal_ee_z)
        return np.linspace(low, high, max(3, self.z_res))

    def _candidate_footprint(self, x, y):
        cx = x + self.target_aabb_offset[0]
        cy = y + self.target_aabb_offset[1]
        hx = self.target_dims[0] / 2.0
        hy = self.target_dims[1] / 2.0
        return cx, cy, hx, hy

    def _footprint_out_of_bounds(self, cx, cy, hx, hy):
        return (
            cx - hx < self.bounds['x_min'] or cx + hx > self.bounds['x_max'] or
            cy - hy < self.bounds['y_min'] or cy + hy > self.bounds['y_max']
        )

    def _overlaps_existing(self, cx, cy, hx, hy, clearance=0.0):
        for fp in self.existing_footprints:
            if (abs(cx - fp['cx']) < hx + fp['hx'] + clearance and
                    abs(cy - fp['cy']) < hy + fp['hy'] + clearance):
                return True, fp['name']
        return False, None

    def _push_path_collision_layer(self):
        layer = np.zeros((self.res, self.res))
        clearance = float(self.push_params.get("push_path_clearance", 0.010))
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                cx, cy, hx, hy = self._candidate_footprint(x, y)
                if self._footprint_out_of_bounds(cx, cy, hx, hy):
                    layer[j, i] = 1.0
                    continue
                hit, _ = self._overlaps_existing(cx, cy, hx, hy, clearance)
                if hit:
                    layer[j, i] = 1.0
                    continue

                candidate = np.array([x, y], dtype=float)
                dist = float(np.linalg.norm(candidate - self.start_xy))
                samples = max(8, int(dist / 0.025))
                blocked = False
                for t in np.linspace(0.0, 1.0, samples):
                    pos = (1.0 - t) * self.start_xy + t * candidate
                    sx, sy, shx, shy = self._candidate_footprint(pos[0], pos[1])
                    hit, _ = self._overlaps_existing(sx, sy, shx, shy, clearance)
                    if hit:
                        blocked = True
                        break
                layer[j, i] = 1.0 if blocked else 0.0
        return layer

    def _push_reachability_layer(self, robot_id, ee_link):
        layer = np.zeros((self.res, self.res))
        robot_base_pos = np.array(p.getBasePositionAndOrientation(robot_id)[0])
        obj_h = self.target_dims[2]
        half_x = self.target_dims[0] / 2.0
        half_y = self.target_dims[1] / 2.0
        _, ee_z = self._nominal_push_heights()

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                candidate = np.array([x, y], dtype=float)
                delta = candidate - self.start_xy
                dist = float(np.linalg.norm(delta))
                if dist < 0.025:
                    layer[j, i] = 0.2
                    continue
                direction = delta / dist
                yaw = np.arctan2(direction[1], direction[0])
                orn = p.getQuaternionFromEuler([np.pi, 0, yaw])
                half_along = abs(direction[0]) * half_x + abs(direction[1]) * half_y
                start_tool = self.start_xy - direction * (half_along + 0.020)
                end_tool = candidate - direction * (
                    half_along + float(self.push_params.get("push_end_margin", 0.0))
                )
                ok = 0
                total = 5
                for t in np.linspace(0.0, 1.0, total):
                    xy = (1.0 - t) * start_tool + t * end_tool
                    base_dist = float(np.linalg.norm(xy - robot_base_pos[:2]))
                    if base_dist > 0.95 or base_dist < 0.18:
                        continue
                    try:
                        increment_metric("ik_calls", phase="acfm")
                        ik = self._calculate_ik_with_limits(
                            robot_id, ee_link,
                            [float(xy[0]), float(xy[1]), float(ee_z)], orn,
                            max_iterations=50,
                            residual_threshold=0.01,
                        )
                        if self._ik_solution_is_valid(
                                robot_id, ee_link,
                                [float(xy[0]), float(xy[1]), float(ee_z)],
                                orn, ik):
                            ok += 1
                    except Exception:
                        pass
                layer[j, i] = ok / total
        return layer

    def _push_reachability_layer_3d(self, robot_id, ee_link):
        layer = np.zeros((self.res, self.res, len(self.z_range)))
        robot_base_pos = np.array(p.getBasePositionAndOrientation(robot_id)[0])
        half_x = self.target_dims[0] / 2.0
        half_y = self.target_dims[1] / 2.0

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                candidate = np.array([x, y], dtype=float)
                delta = candidate - self.start_xy
                dist = float(np.linalg.norm(delta))
                if dist < 0.025:
                    layer[j, i, :] = 0.2
                    continue
                direction = delta / dist
                yaw = np.arctan2(direction[1], direction[0])
                orn = p.getQuaternionFromEuler([np.pi, 0, yaw])
                half_along = abs(direction[0]) * half_x + abs(direction[1]) * half_y
                start_tool = self.start_xy - direction * (half_along + 0.020)
                end_tool = candidate - direction * (
                    half_along + float(self.push_params.get("push_end_margin", 0.0))
                )
                for k, ee_z in enumerate(self.z_range):
                    ok = 0
                    total = 5
                    for t in np.linspace(0.0, 1.0, total):
                        xy = (1.0 - t) * start_tool + t * end_tool
                        base_dist = float(np.linalg.norm(xy - robot_base_pos[:2]))
                        if base_dist > 0.95 or base_dist < 0.18:
                            continue
                        try:
                            increment_metric("ik_calls", phase="acfm")
                            ik = self._calculate_ik_with_limits(
                                robot_id, ee_link,
                                [float(xy[0]), float(xy[1]), float(ee_z)], orn,
                                max_iterations=50,
                                residual_threshold=0.01,
                            )
                            if self._ik_solution_is_valid(
                                    robot_id, ee_link,
                                    [float(xy[0]), float(xy[1]), float(ee_z)],
                                    orn, ik):
                                ok += 1
                        except Exception:
                            pass
                    layer[j, i, k] = ok / total
        return layer

    def _push_contact_height_layer_3d(self):
        layer = np.zeros((self.res, self.res, len(self.z_range)))
        _, nominal_ee_z = self._nominal_push_heights()
        sigma = max(0.006, self.target_dims[2] * 0.18)
        for k, ee_z in enumerate(self.z_range):
            score = float(np.exp(-((ee_z - nominal_ee_z) ** 2) / (2 * sigma ** 2)))
            layer[:, :, k] = score
        return layer

    def _push_goal_progress_layer(self):
        layer = np.zeros((self.res, self.res))
        goal_xy = np.array([self.push_goal["x"], self.push_goal["y"]], dtype=float)
        start_goal_dist = max(float(np.linalg.norm(goal_xy - self.start_xy)), 1e-6)
        target_radius = float(self.push_goal.get("radius", 0.04))
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                candidate = np.array([x, y], dtype=float)
                remaining = float(np.linalg.norm(goal_xy - candidate))
                if remaining <= target_radius:
                    layer[j, i] = 1.0
                else:
                    progress = (start_goal_dist - remaining) / start_goal_dist
                    layer[j, i] = float(np.clip(progress, 0.0, 0.95))
        return layer

    def _push_stability_layer(self):
        layer = np.zeros((self.res, self.res))
        max_push_distance = float(self.push_params.get("push_acfm_max_distance", 0.75))
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                cx, cy, hx, hy = self._candidate_footprint(x, y)
                if self._footprint_out_of_bounds(cx, cy, hx, hy):
                    layer[j, i] = 0.0
                    continue
                dist = float(np.linalg.norm(np.array([x, y]) - self.start_xy))
                distance_score = (
                    1.0 if dist <= max_push_distance
                    else max(0.0, 1.0 - (dist - max_push_distance) / 0.25)
                )
                edge_clear = min(
                    (cx - hx) - self.bounds['x_min'],
                    self.bounds['x_max'] - (cx + hx),
                    (cy - hy) - self.bounds['y_min'],
                    self.bounds['y_max'] - (cy + hy),
                )
                edge_score = float(np.clip(edge_clear / 0.035, 0.0, 1.0))
                layer[j, i] = min(distance_score, edge_score)
        return layer

    def _diagnose_cell(self, x, y):
        if self.grid is None:
            return ["ACFM not computed"]
        i = int(np.argmin(np.abs(self.x_range - x)))
        j = int(np.argmin(np.abs(self.y_range - y)))
        reasons = []
        if self.layers["path_collision"][j, i] > 0.5:
            reasons.append("swept path or final footprint collision")
        if self.layers["reachability"][j, i] < 0.5:
            reasons.append("poor gripper-contact reachability")
        if self.layers["goal_progress"][j, i] < 0.5:
            reasons.append("little progress toward desired target")
        if self.layers["stability"][j, i] < 0.5:
            reasons.append("near table/workspace boundary or too long")
        if self.layers["failure_history"][j, i] > 0.3:
            reasons.append("past failures")
        return reasons or ["clear path, reachable contact, good progress"]

    def get_worst_positions(self, k=3):
        if self.grid is None:
            return []
        flat = np.argsort(self.grid.flatten())[:k]
        results = []
        for idx in flat:
            j, i = divmod(idx, self.res)
            x = float(self.x_range[i])
            y = float(self.y_range[j])
            results.append({
                "x": round(x, 4),
                "y": round(y, 4),
                "score": round(float(self.grid[j, i]), 3),
                "reason": ", ".join(self._diagnose_cell(x, y)),
            })
        return results

    def get_best_positions_3d(self, k=6):
        if self.grid3d is None:
            return [
                {**pos, "z": round(float(self.nominal_ee_z or 0.0), 4)}
                for pos in self.get_best_positions(k)
            ]
        flat = np.argsort(self.grid3d.flatten())[::-1]
        results = []
        chosen_xy = []
        thx = self.target_dims[0] / 2.0 if self.target_dims else 0.0
        thy = self.target_dims[1] / 2.0 if self.target_dims else 0.0
        min_sep_x = thx * 2.0 + 0.01
        min_sep_y = thy * 2.0 + 0.01
        for idx in flat:
            if len(results) >= k:
                break
            yi, xi, zi = np.unravel_index(idx, self.grid3d.shape)
            score = float(self.grid3d[yi, xi, zi])
            if score <= 0.0:
                break
            x = float(self.x_range[xi])
            y = float(self.y_range[yi])
            footprint_cx = x + self.target_aabb_offset[0]
            footprint_cy = y + self.target_aabb_offset[1]
            if any(abs(footprint_cx - px) < min_sep_x and abs(footprint_cy - py) < min_sep_y
                   for px, py in chosen_xy):
                continue
            results.append({
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(float(self.z_range[zi]), 4),
                "score": round(score, 3),
            })
            chosen_xy.append((footprint_cx, footprint_cy))
        return results

    def textualize(self):
        if self.grid is None:
            return "Push ACFM not computed."
        is_3d = self.grid3d is not None
        total = self.res * self.res * len(self.z_range) if is_3d else self.res * self.res
        safe = int(np.sum(self.grid3d > 0.7)) if is_3d else int(np.sum(self.grid > 0.7))
        best = self.get_best_positions_3d(6) if is_3d else self.get_best_positions(6)
        worst = self.get_worst_positions(3)
        gx, gy = self.push_goal["x"], self.push_goal["y"]

        lines = [
            f"=== 3D PUSH FEASIBILITY MAP for pushing '{self.target_name}' ===",
            f"Object size: {[round(d, 3) for d in self.target_dims]}",
            f"Current center: ({self.start_xy[0]:.3f}, {self.start_xy[1]:.3f})",
            (
                f"Desired final target: ({gx:.3f}, {gy:.3f}, "
                f"theta={self.push_goal.get('theta', 0.0):.2f}) "
                f"score={self.direct_target_score:.3f}"
            ),
            (
                f"Gripper/contact z slices evaluated: "
                f"{[round(float(z), 3) for z in self.z_range]}"
                if is_3d else ""
            ),
            (
                f"Nominal gripper z for stable contact: {self.nominal_ee_z:.3f} "
                f"(object contact around z={self.nominal_contact_z:.3f})"
                if self.nominal_ee_z is not None else ""
            ),
            (
                "Push ACFM criteria: swept object footprint must avoid obstacles; "
                "final footprint must be collision-free; gripper-contact waypoints "
                "must be reachable at the selected z height; endpoint should make progress toward the target; "
                "object must remain inside table bounds; past failed endpoints are penalized."
            ),
            f"Safe cells: {safe}/{total} ({safe * 100 // total}%)",
            f"Avg feasibility: {np.mean(self.grid3d if is_3d else self.grid):.2f}",
            "",
            "SAFEST push endpoints / intermediate waypoints (x, y, gripper_z):",
        ]
        lines = [line for line in lines if line != ""]
        for pos in best:
            if "z" in pos:
                lines.append(
                    f"  ({pos['x']:.3f}, {pos['y']:.3f}, z={pos['z']:.3f}) "
                    f"score={pos['score']}"
                )
            else:
                lines.append(f"  ({pos['x']:.3f}, {pos['y']:.3f}) score={pos['score']}")

        if self.direct_target_score < 0.4 and best:
            lines.extend([
                "",
                (
                    "DIRECT TARGET WARNING: direct straight-line push to the final target is risky because "
                    f"{', '.join(self.direct_target_reasons)}."
                ),
                "Use the safest waypoint above first, then push the same object to the desired final target.",
            ])

        lines.append("\nDANGEROUS push endpoints (avoid):")
        for pos in worst:
            lines.append(
                f"  ({pos['x']:.3f}, {pos['y']:.3f}) score={pos['score']} — {pos['reason']}"
            )
        lines.append("=== END 3D PUSH FEASIBILITY MAP ===")
        return "\n".join(lines)

    def render_heatmap(self, save_path=None, planned_positions=None):
        if self.grid is None:
            return None

        from matplotlib.patches import Circle
        x_span = self.bounds['x_max'] - self.bounds['x_min']
        y_span = self.bounds['y_max'] - self.bounds['y_min']
        scale = 9.0 / max(x_span, y_span)
        fig_w = max(5.0, x_span * scale + 2.5)
        fig_h = max(3.5, y_span * scale + 2.0)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'push_feasibility',
            ['#7F1D1D', '#C0392B', '#F39C12', '#F1C40F', '#27AE60', '#145A32'],
            N=256,
        )
        im = ax.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            extent=[
                self.bounds['x_min'], self.bounds['x_max'],
                self.bounds['y_min'], self.bounds['y_max'],
            ],
            origin='lower', aspect='equal', interpolation='bilinear',
        )

        for fp in self.existing_footprints:
            self._draw_rect(
                ax, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=0.0,
                facecolor='#4A235A', edgecolor='white',
                linewidth=1.2, alpha=0.65, zorder=3,
            )
            ax.text(fp['cx'], fp['cy'], fp['name'], ha='center', va='center',
                    fontsize=5.5, color='white', fontweight='bold', zorder=4, clip_on=True)

        thx = self.target_dims[0] / 2.0
        thy = self.target_dims[1] / 2.0
        self._draw_rect(
            ax,
            self.start_xy[0] + self.target_aabb_offset[0],
            self.start_xy[1] + self.target_aabb_offset[1],
            thx, thy, yaw=0.0,
            facecolor='#D6EAF8', edgecolor='black', linewidth=1.4,
            alpha=0.85, zorder=5,
        )
        ax.text(self.start_xy[0], self.start_xy[1], 'start', ha='center',
                va='center', fontsize=6, color='black', fontweight='bold', zorder=6)

        gx, gy = float(self.push_goal["x"]), float(self.push_goal["y"])
        radius = float(self.push_goal.get("radius", 0.04))
        ax.add_patch(Circle((gx, gy), radius, facecolor='#00B894',
                            edgecolor='black', alpha=0.35, zorder=5))
        self._draw_rect(
            ax, gx + self.target_aabb_offset[0], gy + self.target_aabb_offset[1],
            thx, thy, yaw=0.0,
            facecolor='none', edgecolor='#00FFFF',
            linewidth=2.0, zorder=6,
        )
        ax.plot([self.start_xy[0], gx], [self.start_xy[1], gy],
                color='black', linestyle=':', linewidth=1.4, zorder=6)
        ax.text(gx, gy + radius + 0.012, 'desired target', ha='center',
                va='bottom', fontsize=6, color='black', fontweight='bold', zorder=7)

        for bp in self.get_best_positions(4):
            self._draw_rect(
                ax, bp['x'] + self.target_aabb_offset[0],
                bp['y'] + self.target_aabb_offset[1],
                thx, thy, yaw=0.0,
                facecolor='none', edgecolor='white',
                linewidth=1.6, linestyle='--', zorder=8,
            )
            ax.plot(bp['x'], bp['y'], 'w+', markersize=7,
                    markeredgewidth=1.4, zorder=9)
            ax.text(bp['x'], bp['y'] + thy + 0.008, f"{bp['score']:.2f}",
                    ha='center', va='bottom', fontsize=6, color='white',
                    fontweight='bold', zorder=9, clip_on=True)

        if planned_positions:
            for pos in planned_positions:
                px, py = float(pos[0]), float(pos[1])
                self._draw_rect(
                    ax, px + self.target_aabb_offset[0],
                    py + self.target_aabb_offset[1],
                    thx, thy, yaw=0.0,
                    facecolor='#00BFFF', edgecolor='black',
                    linewidth=1.5, alpha=0.75, zorder=10,
                )

        ax.set_xlabel('Push endpoint X (m)')
        ax.set_ylabel('Push endpoint Y (m)')
        ax.set_title(f'Push ACFM: "{self.target_name}"', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Push feasibility')
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(['Blocked', 'Risky', 'Safe', 'Best'])
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64


class ACFM3D(ACFM):
    """3D Action-Conditioned Spatial Feasibility Map for stacking tasks.
    
    Extends the 2D ACFM with a z-axis to evaluate placement at different heights.
    Each (x, y, z) cell gets a feasibility score considering:
    - L1: 3D collision with existing placed objects
    - L2: Reachability (higher placements are harder)
    - L3: Stability (must be supported from below)
    - L4/L5: Same as 2D (disturbance, failure history projected to xy)
    """

    W3D = {
        "collision": 0.30,
        "reachability": 0.20,
        "support": 0.30,  # replaces stability: must have support below
        "disturbance": 0.05,
        "failure_history": 0.15,
    }

    def __init__(self, resolution=15, z_resolution=8, max_z=0.35,
                 rank_candidates=True):
        super().__init__(resolution=resolution,
                         rank_candidates=rank_candidates)
        self.z_res = z_resolution
        self.max_z = max_z
        self.z_range = None
        self.grid3d = None
        self.target_aabb_z_offset = 0.0
        self.support_surfaces = []
        self.allow_ground_support = True
        self.require_configured_support = False

    def compute_3d(self, target_obj_id, target_name, existing_objects,
                   robot_id, ee_link_idx, basket_bounds,
                   failure_history=None, disturbance_history=None,
                   z_values=None, support_surfaces=None,
                   allow_ground_support=True,
                   require_configured_support=False,
                   contact_modes=None, grasp_yaw=0.0):
        """Compute 3D feasibility map for stacking placements."""
        self.target_name = target_name
        self.bounds = basket_bounds
        self.support_surfaces = support_surfaces or []
        self.allow_ground_support = allow_ground_support
        self.require_configured_support = require_configured_support
        requested_modes = tuple(contact_modes or ("top",))
        self.contact_modes = tuple(
            mode for mode in requested_modes if mode in {"top", "front"}
        )
        aabb = p.getAABB(target_obj_id)
        self.target_dims = [aabb[1][i] - aabb[0][i] for i in range(3)]
        body_pos, body_orn = p.getBasePositionAndOrientation(target_obj_id)
        # See the 2-D map: the commanded theta that reproduces this AABB is
        # offset by the grasp yaw, and is unchanged for a square grasp.
        self.target_reference_yaw = (
            float(p.getEulerFromQuaternion(body_orn)[2]) + float(grasp_yaw))
        self.target_aabb_offset = [
            (aabb[0][0] + aabb[1][0]) / 2.0 - body_pos[0],
            (aabb[0][1] + aabb[1][1]) / 2.0 - body_pos[1],
        ]
        self.target_aabb_z_offset = (aabb[0][2] + aabb[1][2]) / 2.0 - body_pos[2]

        self.x_range = np.linspace(basket_bounds['x_min'], basket_bounds['x_max'], self.res)
        self.y_range = np.linspace(basket_bounds['y_min'], basket_bounds['y_max'], self.res)
        if z_values is not None:
            unique_z = sorted({round(float(z), 6) for z in z_values})
            self.z_range = np.array(unique_z, dtype=float)
            self.z_res = len(self.z_range)
            if self.z_res:
                self.max_z = max(self.max_z, float(np.max(self.z_range)))
        else:
            self.z_range = np.linspace(0.01, self.max_z, self.z_res)

        # Cache placed-object footprints (x, y, z) for visualization
        self.existing_footprints = []
        for _name, _oid in existing_objects.items():
            if _oid == target_obj_id:
                continue
            _aabb = p.getAABB(_oid)
            _, _orn = p.getBasePositionAndOrientation(_oid)
            _yaw = p.getEulerFromQuaternion(_orn)[2]
            self.existing_footprints.append({
                'name': _name,
                'cx': (_aabb[0][0] + _aabb[1][0]) / 2.0,
                'cy': (_aabb[0][1] + _aabb[1][1]) / 2.0,
                'cz': (_aabb[0][2] + _aabb[1][2]) / 2.0,
                'hx': (_aabb[1][0] - _aabb[0][0]) / 2.0,
                'hy': (_aabb[1][1] - _aabb[0][1]) / 2.0,
                'hz': (_aabb[1][2] - _aabb[0][2]) / 2.0,
                'yaw': _yaw,
                'geometry_yaw': 0.0,
                'z_min': _aabb[0][2],
                'z_max': _aabb[1][2],
            })

        # Cheap hard masks first: they gate the expensive IK sweep below.
        L1 = self._collision_layer_3d(target_obj_id, existing_objects)
        L3 = self._support_layer(target_obj_id, existing_objects)
        _bounds_2d = self._bounds_mask_2d()
        _bounds_3d = np.repeat(
            _bounds_2d[:, :, np.newaxis], self.z_res, axis=2)
        _skip = np.logical_or.reduce((
            ~_bounds_3d, L1 >= 1.0, L3 <= 0.0))
        L2 = self._reachability_layer_3d(
            robot_id, ee_link_idx, skip_mask=_skip)
        L4 = self._disturbance_layer_projected(disturbance_history)
        L5 = self._failure_layer_projected(failure_history)

        bounds_2d = _bounds_2d
        bounds_mask = _bounds_3d
        collision_mask = np.logical_or(~bounds_mask, L1 < 1.0)
        reachability_mask = L2 > 0.0
        # Partial footprint contact is a soft support margin, not valid hard
        # support.  Require at least the implementation's 50% coverage tier.
        support_mask = L3 >= 0.5
        contact_mask = np.full_like(
            bounds_mask, bool(self.contact_modes), dtype=bool)
        self.grid3d = self._compose_feasible_scores(
            {
                "bounds": bounds_mask,
                "collision": collision_mask,
                "reachability": reachability_mask,
                "support": support_mask,
                "valid_contact": contact_mask,
            },
            {
                "collision": 1.0 - L1,
                "reachability": L2,
                "support": L3,
                "disturbance": 1.0 - L4,
                "failure_history": 1.0 - L5,
            },
            self.W3D,
        )

        # Also compute the 2D projection (max over z) for compatibility
        self.grid = np.max(self.grid3d, axis=2)
        self.layers = {
            "collision_3d": L1,
            "reachability_3d": L2,
            "support": L3,
            "disturbance": L4,
            "failure_history": L5,
        }
        return self.grid3d

    def _collision_layer_3d(self, target_obj_id, existing_objects):
        """3D collision: check AABB overlap in x, y, AND z."""
        layer = np.zeros((self.res, self.res, self.z_res))
        t_half = [self.target_dims[k] / 2 for k in range(3)]

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                # Boundary check (xy)
                cx = x + self.target_aabb_offset[0]
                cy = y + self.target_aabb_offset[1]
                obj_xmin = cx - t_half[0]
                obj_xmax = cx + t_half[0]
                obj_ymin = cy - t_half[1]
                obj_ymax = cy + t_half[1]
                out_of_bounds = (
                    obj_xmin < self.bounds['x_min'] or
                    obj_xmax > self.bounds['x_max'] or
                    obj_ymin < self.bounds['y_min'] or
                    obj_ymax > self.bounds['y_max']
                )

                for k, z in enumerate(self.z_range):
                    if out_of_bounds:
                        layer[j, i, k] = 1.0
                        continue
                    cz = z + self.target_aabb_z_offset
                    obj_zmin = cz - t_half[2]
                    obj_zmax = cz + t_half[2]

                    collision = False
                    for name, oid in existing_objects.items():
                        if oid == target_obj_id:
                            continue
                        oaabb = p.getAABB(oid)
                        # 3D AABB overlap
                        z_overlap = (
                            min(obj_zmax, oaabb[1][2]) -
                            max(obj_zmin, oaabb[0][2])
                        )
                        if (obj_xmin < oaabb[1][0] and obj_xmax > oaabb[0][0] and
                            obj_ymin < oaabb[1][1] and obj_ymax > oaabb[0][1] and
                            z_overlap > 0.003):
                            collision = True
                            break
                    layer[j, i, k] = 1.0 if collision else 0.0
        return layer

    def _reachability_layer_3d(self, robot_id, ee_link, skip_mask=None):
        """Verified endpoint IK with distance/height used only as soft quality.

        ``skip_mask`` marks cells already rejected by a cheaper hard mask; see
        the 2-D counterpart.  In 3-D the support mask rejects most of the
        volume -- a placement needs a surface under it -- so gating the IK
        sweep on it removes the bulk of the calls.
        """
        layer = np.zeros((self.res, self.res, self.z_res))
        robot_base_pos = np.array(p.getBasePositionAndOrientation(robot_id)[0])

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    if skip_mask is not None and skip_mask[j, i, k]:
                        continue
                    mode_scores = []
                    for mode in self.contact_modes:
                        target_pos, orientations = self._contact_ee_poses(
                            x, y, z, mode)
                        reached = self._first_reachable_orientation(
                            robot_id, ee_link, target_pos, orientations)
                        if reached is None:
                            continue
                        dist_3d = np.linalg.norm(
                            np.asarray(target_pos) - robot_base_pos)
                        if dist_3d > 0.85:
                            mode_scores.append(0.1)
                        elif dist_3d > 0.75:
                            mode_scores.append(0.4)
                        elif z > 0.3:
                            mode_scores.append(0.5)
                        elif dist_3d < 0.25:
                            mode_scores.append(0.3)
                        else:
                            mode_scores.append(1.0)
                    layer[j, i, k] = max(mode_scores, default=0.0)
        return layer

    def _support_layer(self, target_obj_id, existing_objects):
        """Key for stacking: object must be supported from below.
        
        Score = 1.0 if directly on table/basket floor (z ~ 0)
        Score = 1.0 if top surface of existing object is directly below
        Score = 0.0 if floating in air (no support)
        """
        layer = np.zeros((self.res, self.res, self.z_res))
        t_half = [self.target_dims[k] / 2 for k in range(3)]
        support_tol = 0.02  # 2cm tolerance for support contact

        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    cx = x + self.target_aabb_offset[0]
                    cy = y + self.target_aabb_offset[1]
                    cz = z + self.target_aabb_z_offset
                    obj_bottom = cz - t_half[2]

                    # On the ground/basket floor
                    if self.allow_ground_support and obj_bottom < support_tol:
                        layer[j, i, k] = 1.0
                        continue

                    # On configured support surfaces, e.g. shelf boards.
                    for surface in self.support_surfaces:
                        sx_min = surface.get('x_min', self.bounds['x_min'])
                        sx_max = surface.get('x_max', self.bounds['x_max'])
                        sy_min = surface.get('y_min', self.bounds['y_min'])
                        sy_max = surface.get('y_max', self.bounds['y_max'])
                        surface_z = surface['z']
                        if abs(obj_bottom - surface_z) > support_tol:
                            continue
                        overlap_x = max(
                            0.0,
                            min(cx + t_half[0], sx_max)
                            - max(cx - t_half[0], sx_min),
                        )
                        overlap_y = max(
                            0.0,
                            min(cy + t_half[1], sy_max)
                            - max(cy - t_half[1], sy_min),
                        )
                        target_area = self.target_dims[0] * self.target_dims[1]
                        coverage = (
                            overlap_x * overlap_y / target_area
                            if target_area > 0.0 else 0.0
                        )
                        # Apply the same support tiers used for object-backed
                        # surfaces below.  Mild overhang is not unsupported;
                        # the hard mask still requires the >=50% tier.
                        if coverage >= 0.8:
                            layer[j, i, k] = 1.0
                        elif coverage >= 0.5:
                            layer[j, i, k] = max(layer[j, i, k], 0.7)
                        elif coverage > 0.2:
                            layer[j, i, k] = max(layer[j, i, k], 0.3)
                    if layer[j, i, k] > 0.0:
                        continue

                    if self.require_configured_support and self.support_surfaces:
                        layer[j, i, k] = 0.0
                        continue

                    # Check if any existing object provides support below
                    best_support = 0.0
                    for name, oid in existing_objects.items():
                        if oid == target_obj_id:
                            continue
                        oaabb = p.getAABB(oid)
                        o_top = oaabb[1][2]
                        o_xmin, o_xmax = oaabb[0][0], oaabb[1][0]
                        o_ymin, o_ymax = oaabb[0][1], oaabb[1][1]

                        # Check vertical alignment: obj bottom near existing top
                        z_gap = abs(obj_bottom - o_top)
                        if z_gap > support_tol:
                            continue

                        # Check horizontal overlap: target footprint must overlap support
                        overlap_x = max(0, min(cx + t_half[0], o_xmax) - max(cx - t_half[0], o_xmin))
                        overlap_y = max(0, min(cy + t_half[1], o_ymax) - max(cy - t_half[1], o_ymin))
                        overlap_area = overlap_x * overlap_y
                        target_area = self.target_dims[0] * self.target_dims[1]

                        if target_area > 0:
                            coverage = overlap_area / target_area
                            # Need >=50% coverage for good support
                            if coverage >= 0.8:
                                best_support = 1.0
                            elif coverage >= 0.5:
                                best_support = max(best_support, 0.7)
                            elif coverage > 0.2:
                                best_support = max(best_support, 0.3)

                    layer[j, i, k] = best_support
        return layer

    def support_score_at(self, x, y, z):
        """Evaluate support at exact controller parameters, without grid aliasing."""
        support_tol = 0.02
        t_half = [self.target_dims[index] / 2.0 for index in range(3)]
        cx = float(x) + self.target_aabb_offset[0]
        cy = float(y) + self.target_aabb_offset[1]
        cz = float(z) + self.target_aabb_z_offset
        obj_bottom = cz - t_half[2]

        if self.allow_ground_support and obj_bottom < support_tol:
            return 1.0

        def coverage_score(x_min, x_max, y_min, y_max):
            overlap_x = max(
                0.0,
                min(cx + t_half[0], x_max)
                - max(cx - t_half[0], x_min),
            )
            overlap_y = max(
                0.0,
                min(cy + t_half[1], y_max)
                - max(cy - t_half[1], y_min),
            )
            target_area = self.target_dims[0] * self.target_dims[1]
            coverage = (
                overlap_x * overlap_y / target_area
                if target_area > 0.0 else 0.0
            )
            if coverage >= 0.8:
                return 1.0
            if coverage >= 0.5:
                return 0.7
            if coverage > 0.2:
                return 0.3
            return 0.0

        configured_score = 0.0
        for surface in self.support_surfaces:
            if abs(obj_bottom - float(surface["z"])) > support_tol:
                continue
            configured_score = max(
                configured_score,
                coverage_score(
                    float(surface.get("x_min", self.bounds["x_min"])),
                    float(surface.get("x_max", self.bounds["x_max"])),
                    float(surface.get("y_min", self.bounds["y_min"])),
                    float(surface.get("y_max", self.bounds["y_max"])),
                ),
            )
        if configured_score > 0.0:
            return configured_score
        if self.require_configured_support and self.support_surfaces:
            return 0.0

        object_score = 0.0
        for footprint in self.existing_footprints:
            if abs(obj_bottom - float(footprint["z_max"])) > support_tol:
                continue
            object_score = max(
                object_score,
                coverage_score(
                    footprint["cx"] - footprint["hx"],
                    footprint["cx"] + footprint["hx"],
                    footprint["cy"] - footprint["hy"],
                    footprint["cy"] + footprint["hy"],
                ),
            )
        return object_score

    def _disturbance_layer_projected(self, history):
        """Project 2D disturbance history onto 3D (same for all z)."""
        layer2d = self._disturbance_layer(history)
        return np.repeat(layer2d[:, :, np.newaxis], self.z_res, axis=2)

    def _failure_layer_projected(self, history):
        """Project 2D failure history onto 3D (same for all z)."""
        layer2d = self._failure_layer(history)
        return np.repeat(layer2d[:, :, np.newaxis], self.z_res, axis=2)

    def get_best_positions_3d(self, k=5):
        """Return k positions with highest 3D feasibility."""
        if self.grid3d is None:
            return []
        flat = np.argsort(self.grid3d.flatten())[::-1]
        results = []
        for idx in flat:
            yi, xi, zi = np.unravel_index(idx, self.grid3d.shape)
            score = float(self.grid3d[yi, xi, zi])
            if score <= 0.0:
                break
            if self.feasible_mask is not None and not self.feasible_mask[yi, xi, zi]:
                continue
            results.append({
                "x": round(float(self.x_range[xi]), 4),
                "y": round(float(self.y_range[yi]), 4),
                "z": round(float(self.z_range[zi]), 4),
                "score": round(score, 3),
            })
            if len(results) >= k:
                break
        return results

    def get_parameter_candidates(self, k=8, yaw_values=None,
                                 contact_modes=None, z=None):
        """Return ID-addressed feasible (x, y, z, yaw, contact) rows."""
        if self.grid3d is None or self.feasible_mask is None or k <= 0:
            return []
        if yaw_values is None:
            width, length = self.target_dims[:2]
            rectangular = max(width, length) / max(min(width, length), 1e-9) > 1.10
            yaw_values = (0.0, np.pi / 2.0) if rectangular else (0.0,)
        modes = tuple(contact_modes or self.contact_modes)
        modes = tuple(mode for mode in modes if mode in self.contact_modes)
        if not modes:
            return []

        rows = []
        seen = set()
        for index in np.argsort(self.grid3d.flatten())[::-1]:
            yi, xi, zi = np.unravel_index(int(index), self.grid3d.shape)
            if not self.feasible_mask[yi, xi, zi]:
                continue
            score = float(self.grid3d[yi, xi, zi])
            if score <= 0.0:
                continue
            x = float(self.x_range[xi])
            y = float(self.y_range[yi])
            z_value = float(self.z_range[zi])
            for yaw in yaw_values:
                yaw = float(yaw)
                if not self._oriented_candidate_is_endpoint_feasible(
                        x, y, yaw, z=z_value):
                    continue
                for mode in modes:
                    if mode == "front" and not (
                            self._front_approach_envelope_is_clear(
                                x, y, z_value)):
                        continue
                    key = (
                        round(x, 6), round(y, 6), round(z_value, 6),
                        round(yaw, 6), mode,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "candidate_id": "",
                        "object": self.target_name,
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "z": round(z_value, 4),
                        "theta": round(yaw, 4),
                        "grasp_mode": (
                            mode if mode in {"top", "front"} else None),
                        "contact_mode": mode,
                        "score": round(score, 4),
                        "hard_feasible": True,
                        "hard_constraints": {
                            name: True for name in self.hard_masks
                        },
                    })
            if len(rows) >= k * 4:
                break

        rows.sort(key=lambda row: (
            -row["score"], row["x"], row["y"], row["z"],
            row["theta"], row["contact_mode"]))
        rows = rows[:k]
        for idx, row in enumerate(rows, start=1):
            row["candidate_id"] = f"C{idx:04d}"
        self.parameter_candidates = rows
        return [dict(row) for row in rows]

    def _z_support_note(self, z):
        obj_bottom = z + self.target_aabb_z_offset - self.target_dims[2] / 2.0
        if self.allow_ground_support and obj_bottom < 0.03:
            return " (on table/floor)"
        if self.support_surfaces:
            nearest = min(
                self.support_surfaces,
                key=lambda s: abs(obj_bottom - s['z']),
            )
            if abs(obj_bottom - nearest['z']) <= 0.03:
                label = nearest.get('label', 'support surface')
                return f" (on {label}, surface z={nearest['z']:.3f})"
        if z < 0.02:
            return " (on floor)"
        if z > 0.05:
            return " (stacked/on elevated support)"
        return ""

    def textualize(self):
        """Convert 3D ACFM to text for LLM prompt."""
        if self.grid3d is None:
            return super().textualize()

        total = self.res * self.res * self.z_res
        safe = int(np.sum(self.grid3d > 0.7))
        best = self.get_parameter_candidates(5)

        lines = [
            f"=== 3D SPATIAL FEASIBILITY MAP for placing '{self.target_name}' ===",
            f"Object size: {[round(d, 3) for d in self.target_dims]}",
            f"Z slices evaluated: {[round(float(z), 3) for z in self.z_range]}",
            f"Safe cells: {safe}/{total} ({safe * 100 // total}%)",
            f"Avg feasibility: {np.mean(self.grid3d):.2f}",
            "",
            "FEASIBLE CANDIDATE ROWS (select candidate_id; do not copy coordinates):",
        ]
        for pos in best:
            support_note = self._z_support_note(pos['z'])
            lines.append(
                f"  {pos['candidate_id']} | x={pos['x']:.3f} | "
                f"y={pos['y']:.3f} | z={pos['z']:.3f} | "
                f"yaw={pos['theta']:.4f} | grasp/contact={pos['contact_mode']} | "
                f"score={pos['score']}{support_note}"
            )

        # Also show 2D projection summary
        safe_2d = int(np.sum(self.grid > 0.7))
        lines.append(f"\n2D footprint: {safe_2d}/{self.res**2} safe cells")
        lines.append("=== END 3D FEASIBILITY MAP ===")
        return "\n".join(lines)

    def render_heatmap_3d(self, save_path=None):
        """
        Render multi-panel 3D ACFM visualization:
        - Top row: z-slice heatmaps at multiple heights with cross-section footprints
        - Bottom-left: 2D projection (max over z) with all footprints
        - Bottom-center: support layer
        - Bottom-right: collision layer
        Returns base64 PNG.
        """
        if self.grid3d is None:
            return self.render_heatmap(save_path=save_path)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'feasibility',
            ['#C0392B', '#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#1E8449'],
            N=256,
        )

        # Pick z-slice indices to show (evenly spaced, up to 6)
        n_slices = min(self.z_res, 6)
        slice_indices = np.linspace(0, self.z_res - 1, n_slices, dtype=int)

        nrows = 2
        ncols = max(n_slices, 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 8), dpi=120)
        w_cm = round(self.target_dims[0] * 100, 1)
        l_cm = round(self.target_dims[1] * 100, 1)
        h_cm = round(self.target_dims[2] * 100, 1)
        fig.suptitle(
            f'3D ACFM: "{self.target_name}"  '
            f'[{w_cm}\u00d7{l_cm}\u00d7{h_cm} cm  (W\u00d7L\u00d7H)]',
            fontsize=13, fontweight='bold',
        )

        extent = [
            self.bounds['x_min'], self.bounds['x_max'],
            self.bounds['y_min'], self.bounds['y_max'],
        ]
        thx = self.target_dims[0] / 2.0
        thy = self.target_dims[1] / 2.0

        # Top row: z-slice heatmaps
        for col, zi in enumerate(slice_indices):
            ax = axes[0, col]
            z_val = self.z_range[zi]
            ax.imshow(
                self.grid3d[:, :, zi], cmap=cmap, vmin=0, vmax=1,
                extent=extent, origin='lower', aspect='equal',
                interpolation='bilinear',
            )
            ax.set_title(f'z = {z_val:.3f} m', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')

            # Existing objects: draw cross-section if this z-slice intersects the object
            for fp in self.existing_footprints:
                if fp['z_min'] <= z_val <= fp['z_max']:
                    self._draw_rect(
                        ax, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=fp['yaw'],
                        facecolor='#444444', edgecolor='white',
                        linewidth=1.0, alpha=0.55, zorder=3,
                    )
                    ax.text(fp['cx'], fp['cy'], fp['name'],
                            ha='center', va='center', fontsize=4.5,
                            color='white', fontweight='bold', zorder=4, clip_on=True)

            # Best target position in this z-slice: dashed rectangle
            slice_data = self.grid3d[:, :, zi]
            by, bx = np.unravel_index(np.argmax(slice_data), slice_data.shape)
            bx_pos = self.x_range[bx]
            by_pos = self.y_range[by]
            self._draw_rect(
                ax, bx_pos, by_pos, thx, thy, yaw=0.0,
                facecolor='none', edgecolor='white',
                linewidth=1.4, linestyle='--', zorder=5,
            )
            ax.plot(bx_pos, by_pos, 'w+', markersize=7,
                    markeredgewidth=1.2, zorder=6)

        # Hide unused columns in top row
        for col in range(len(slice_indices), ncols):
            axes[0, col].set_visible(False)

        # Bottom-left: 2D projection (max over z) with all footprints
        ax_proj = axes[1, 0]
        ax_proj.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            extent=extent, origin='lower', aspect='equal',
            interpolation='bilinear',
        )
        ax_proj.set_title('2D Projection (max over z)', fontsize=10)
        ax_proj.set_xlabel('X (m)')
        ax_proj.set_ylabel('Y (m)')

        for fp in self.existing_footprints:
            self._draw_rect(
                ax_proj, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=fp['yaw'],
                facecolor='#444444', edgecolor='white',
                linewidth=1.2, alpha=0.6, zorder=3,
            )
            ax_proj.text(fp['cx'], fp['cy'], fp['name'],
                         ha='center', va='center', fontsize=5,
                         color='white', fontweight='bold', zorder=4, clip_on=True)

        for bp in self.get_best_positions_3d(3):
            self._draw_rect(
                ax_proj, bp['x'], bp['y'], thx, thy, yaw=0.0,
                facecolor='none', edgecolor='white',
                linewidth=1.5, linestyle='--', zorder=5,
            )
            ax_proj.plot(bp['x'], bp['y'], 'w+', markersize=7,
                         markeredgewidth=1.2, zorder=6)
            ax_proj.text(
                bp['x'], bp['y'] + thy + 0.008,
                f"z={bp['z']:.2f}  s={bp['score']:.2f}",
                ha='center', va='bottom', fontsize=5,
                color='white', fontweight='bold', zorder=6, clip_on=True,
            )

        # Bottom-center: support layer (critical for stacking)
        if 'support' in self.layers:
            ax_sup = axes[1, 1]
            # Show support at the z-slice with best overall support
            support_by_z = np.mean(self.layers['support'], axis=(0, 1))
            best_z_idx = np.argmax(support_by_z)
            im_sup = ax_sup.imshow(
                self.layers['support'][:, :, best_z_idx],
                cmap='YlGn', vmin=0, vmax=1,
                extent=extent, origin='lower', aspect='equal',
                interpolation='bilinear',
            )
            ax_sup.set_title(
                f'Support Layer (z={self.z_range[best_z_idx]:.3f}m)', fontsize=10)
            ax_sup.set_xlabel('X (m)')
            ax_sup.set_ylabel('Y (m)')
            plt.colorbar(im_sup, ax=ax_sup, shrink=0.7)

        # Bottom-right: collision layer
        if 'collision_3d' in self.layers:
            ax_col = axes[1, 2]
            col_proj = np.min(self.layers['collision_3d'], axis=2)
            im_col = ax_col.imshow(
                col_proj, cmap='RdYlGn_r', vmin=0, vmax=1,
                extent=extent, origin='lower', aspect='equal',
                interpolation='bilinear',
            )
            ax_col.set_title('Collision (min over z)', fontsize=10)
            ax_col.set_xlabel('X (m)')
            ax_col.set_ylabel('Y (m)')
            plt.colorbar(im_col, ax=ax_col, shrink=0.7)

        # Hide unused columns in bottom row
        for col in range(3, ncols):
            axes[1, col].set_visible(False)

        # Global colorbar for feasibility
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
            ax=axes[0, :].tolist(), shrink=0.6, pad=0.02, label='Feasibility',
        )
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(['Danger', 'Risky', 'Safe', 'Optimal'])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64


class HanoiACFM:
    """Discrete 3D feasibility map for Tower of Hanoi peg moves."""

    def __init__(self, resolution=3):
        self.res = resolution
        self.env = None
        self.peg_state = {}
        self.peg_order = []
        self.disk_order = []
        self.disk_widths = {}
        self.disk_dims = {}
        self.legal_moves = []
        self.illegal_moves = []
        self.bounds = None
        self.target_name = "hanoi_state"
        self.grid = None
        self.grid3d = None
        self.x_range = None
        self.y_range = None
        self.z_range = None

    def compute_hanoi(self, env):
        """Compute legal peg-to-peg moves and their 3D target poses."""
        self.env = env
        self.peg_order = [
            peg for peg in ("A", "B", "C") if peg in env._pegs
        ] + [
            peg for peg in env._pegs if peg not in ("A", "B", "C")
        ]
        raw_state = env._get_peg_state()
        self.peg_state = {
            peg: list(raw_state.get(peg, []))
            for peg in self.peg_order
        }
        self.disk_order = list(env._disks_ordered)
        self.disk_widths = dict(getattr(env, "_disk_widths", {}))
        self.disk_dims = {}
        self.legal_moves = []
        self.illegal_moves = []

        for disk_name in self.disk_order:
            if disk_name not in env.objects:
                continue
            aabb = p.getAABB(env.objects[disk_name])
            self.disk_dims[disk_name] = [
                aabb[1][0] - aabb[0][0],
                aabb[1][1] - aabb[0][1],
                aabb[1][2] - aabb[0][2],
            ]

        xs = [env._pegs[p]["x"] for p in self.peg_order]
        ys = [env._pegs[p]["y"] for p in self.peg_order]
        pad = 0.18
        self.bounds = {
            "x_min": min(xs) - pad,
            "x_max": max(xs) + pad,
            "y_min": min(ys) - pad,
            "y_max": max(ys) + pad,
        }

        for src in self.peg_order:
            stack = self.peg_state.get(src, [])
            if not stack:
                continue
            top_disk = stack[-1]
            for dst in self.peg_order:
                if dst == src:
                    continue
                dst_stack = self.peg_state.get(dst, [])
                dst_top = dst_stack[-1] if dst_stack else None
                if dst_top is None or self._is_smaller(top_disk, dst_top):
                    x = float(env._pegs[dst]["x"])
                    y = float(env._pegs[dst]["y"])
                    z = float(env._next_place_z(dst, raw_state, top_disk) + 0.003)
                    self.legal_moves.append({
                        "disk": top_disk,
                        "source": src,
                        "target": dst,
                        "target_top": dst_top,
                        "x": x,
                        "y": y,
                        "z": z,
                        "theta": 0.0,
                        "score": 1.0,
                    })
                else:
                    self.illegal_moves.append({
                        "disk": top_disk,
                        "source": src,
                        "target": dst,
                        "target_top": dst_top,
                        "reason": (
                            f"{top_disk} is larger than top disk {dst_top} on peg {dst}"
                        ),
                    })

            for buried_disk in stack[:-1]:
                self.illegal_moves.append({
                    "disk": buried_disk,
                    "source": src,
                    "target": "*",
                    "target_top": top_disk,
                    "reason": (
                        f"{buried_disk} is not the top disk on peg {src}; "
                        f"{top_disk} must move first"
                    ),
                })

        self._compute_state_grid(env)
        return self.legal_moves

    def _compute_state_grid(self, env):
        xy_res = max(12, int(self.res))
        z_res = max(6, min(14, xy_res // 2))
        next_z_values = [
            env._next_place_z(peg, self.peg_state) + 0.003
            for peg in self.peg_order
        ]
        max_disk_h = max(
            (dims[2] for dims in self.disk_dims.values()),
            default=0.025,
        )
        self.x_range = np.linspace(self.bounds["x_min"], self.bounds["x_max"], xy_res)
        self.y_range = np.linspace(self.bounds["y_min"], self.bounds["y_max"], xy_res)
        z_hi = max(next_z_values or [0.04]) + max_disk_h * 2.0
        self.z_range = np.linspace(0.005, z_hi, z_res)
        self.grid3d = np.zeros((len(self.y_range), len(self.x_range), len(self.z_range)))

        xy_sigma = 0.035
        z_sigma = max(0.010, max_disk_h * 0.55)
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    score = 0.0
                    for move in self.legal_moves:
                        dxy = float(np.hypot(x - move["x"], y - move["y"]))
                        dz = abs(float(z - move["z"]))
                        xy_score = np.exp(-(dxy ** 2) / (2.0 * xy_sigma ** 2))
                        z_score = np.exp(-(dz ** 2) / (2.0 * z_sigma ** 2))
                        score = max(score, self._move_score(move) * xy_score * z_score)
                    self.grid3d[j, i, k] = float(np.clip(score, 0.0, 1.0))
        self.grid = np.max(self.grid3d, axis=2)

    def _rank(self, disk_name):
        if disk_name in self.disk_order:
            return self.disk_order.index(disk_name)
        return -1

    def _is_smaller(self, disk_name, other_name):
        disk_rank = self._rank(disk_name)
        other_rank = self._rank(other_name)
        if disk_rank >= 0 and other_rank >= 0:
            return disk_rank > other_rank
        return self.disk_widths.get(disk_name, 0.0) < self.disk_widths.get(other_name, 0.0)

    def _move_score(self, move):
        dst_stack = self.peg_state.get(move["target"], [])
        if not dst_stack:
            return 1.0
        return 0.95

    def _disk_center(self, disk_name):
        if self.env is None or disk_name not in self.env.objects:
            return np.array([0.0, 0.0, 0.0])
        return np.array(p.getBasePositionAndOrientation(self.env.objects[disk_name])[0])

    def _nearest_peg(self, x, y):
        if self.env is None:
            return None, float("inf")
        nearest = None
        nearest_dist = float("inf")
        for peg in self.peg_order:
            peg_cfg = self.env._pegs[peg]
            dist = float(np.hypot(x - peg_cfg["x"], y - peg_cfg["y"]))
            if dist < nearest_dist:
                nearest = peg
                nearest_dist = dist
        return nearest, nearest_dist

    def get_score_at(self, x, y):
        if self.grid is not None and self.x_range is not None and self.y_range is not None:
            i = int(np.argmin(np.abs(self.x_range - x)))
            j = int(np.argmin(np.abs(self.y_range - y)))
            return float(self.grid[j, i])
        peg, dist = self._nearest_peg(x, y)
        if peg is None:
            return 0.0
        tol = 0.08
        if self.env is not None:
            tol = min(0.13, max(0.05, getattr(self.env, "PEG_RADIUS", 0.13)))
        if dist > tol:
            return 0.0
        if any(move["target"] == peg for move in self.legal_moves):
            return 1.0
        return 0.15

    def get_best_positions(self, k=5):
        results = []
        for move in sorted(self.legal_moves, key=lambda m: (-self._move_score(m), m["target"])):
            results.append({
                "x": round(move["x"], 4),
                "y": round(move["y"], 4),
                "z": round(move["z"], 4),
                "score": round(self._move_score(move), 3),
                "disk": move["disk"],
                "move": f"{move['source']}->{move['target']}",
            })
            if len(results) >= k:
                break
        return results

    def get_best_positions_3d(self, k=5):
        return self.get_best_positions(k)

    def textualize(self):
        if self.env is None:
            return "Hanoi ACFM not computed."

        lines = [
            "=== 3D HANOI FEASIBILITY MAP ===",
            (
                "Criteria: only the top disk on a peg may move; a larger disk "
                "cannot be placed on a smaller disk; target x/y must be the exact "
                "peg center; target z is the next stack height."
            ),
            "Current peg stacks (bottom -> top):",
        ]
        for peg in self.peg_order:
            peg_cfg = self.env._pegs[peg]
            stack = self.peg_state.get(peg, [])
            stack_text = " < ".join(stack) if stack else "EMPTY"
            next_z = self.env._next_place_z(peg, self.peg_state) + 0.003
            lines.append(
                f"  Peg {peg}: ({peg_cfg['x']:.3f}, {peg_cfg['y']:.3f}), "
                f"next z={next_z:.4f}, stack={stack_text}"
            )

        lines.extend(["", "LEGAL 3D moves / safest placements:"])
        if not self.legal_moves:
            lines.append("  No legal moves found from the current peg state.")
        for move in self.legal_moves:
            support = (
                f"onto {move['target_top']}" if move["target_top"] else "onto empty peg"
            )
            lines.append(
                f"  {move['disk']}: {move['source']} -> {move['target']} ({support}); "
                f"place(['{move['disk']}'], "
                f"{{'x': {move['x']:.3f}, 'y': {move['y']:.3f}, "
                f"'z': {move['z']:.4f}, 'theta': 0.0}}) "
                f"score={self._move_score(move):.2f}"
            )

        if self.illegal_moves:
            lines.extend(["", "ILLEGAL / dangerous alternatives to avoid:"])
            for move in self.illegal_moves[:8]:
                target = move["target"]
                target_text = target if target != "*" else "any target"
                lines.append(
                    f"  {move['disk']} from {move['source']} to {target_text}: "
                    f"{move['reason']}"
                )
        lines.append("=== END 3D HANOI FEASIBILITY MAP ===")
        return "\n".join(lines)

    def render_3d_volume(self, save_path=None):
        if self.env is None:
            return None

        from matplotlib.patches import Circle
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        def _box_faces(cx, cy, cz, hx, hy, hz):
            x0, x1 = cx - hx, cx + hx
            y0, y1 = cy - hy, cy + hy
            z0, z1 = cz - hz, cz + hz
            return [
                [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
                [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
                [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
                [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
            ]

        def _body_color(disk_name):
            try:
                visual = p.getVisualShapeData(self.env.objects[disk_name])
                if visual:
                    rgba = visual[0][7]
                    return rgba[:3] + (min(max(rgba[3], 0.35), 0.95),)
            except Exception:
                pass
            idx = max(0, self._rank(disk_name))
            palette = plt.cm.viridis(np.linspace(0.2, 0.85, max(2, len(self.disk_order))))
            return palette[idx % len(palette)]

        fig = plt.figure(figsize=(16, 7), dpi=150)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        z_top = 0.12
        for peg in self.peg_order:
            peg_cfg = self.env._pegs[peg]
            px, py = float(peg_cfg["x"]), float(peg_cfg["y"])
            ax3d.plot([px, px], [py, py], [0.0, z_top], color="#222222", linewidth=2.0)
            ax3d.text(px, py, z_top + 0.005, f"Peg {peg}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")
            ax2d.add_patch(
                Circle((px, py), 0.105, facecolor="#E8F4FD", edgecolor="#2C3E50",
                       linewidth=1.0, alpha=0.65)
            )
            ax2d.text(px, py - 0.13, f"Peg {peg}", ha="center", va="top",
                      fontsize=8, fontweight="bold")

            for disk_name in self.peg_state.get(peg, []):
                if disk_name not in self.env.objects:
                    continue
                aabb = p.getAABB(self.env.objects[disk_name])
                cx = (aabb[0][0] + aabb[1][0]) / 2.0
                cy = (aabb[0][1] + aabb[1][1]) / 2.0
                cz = (aabb[0][2] + aabb[1][2]) / 2.0
                hx = (aabb[1][0] - aabb[0][0]) / 2.0
                hy = (aabb[1][1] - aabb[0][1]) / 2.0
                hz = (aabb[1][2] - aabb[0][2]) / 2.0
                z_top = max(z_top, cz + hz + 0.04)
                color = _body_color(disk_name)
                faces = _box_faces(cx, cy, cz, hx, hy, hz)
                poly = Poly3DCollection(
                    faces, facecolor=color, edgecolor="black",
                    linewidth=0.4, alpha=color[3]
                )
                ax3d.add_collection3d(poly)
                ax3d.text(cx, cy, cz + hz + 0.002, disk_name.replace("disk_", "d"),
                          ha="center", va="bottom", fontsize=6)

                radius = max(hx, hy)
                ax2d.add_patch(
                    Circle((cx, cy), radius, facecolor=color,
                           edgecolor="black", linewidth=0.6, alpha=0.75)
                )
                if disk_name == self.peg_state.get(peg, [])[-1]:
                    ax2d.text(cx, cy, disk_name.replace("disk_", "d"),
                              ha="center", va="center", fontsize=6,
                              fontweight="bold", color="black")

        for move in self.legal_moves:
            src_xy = self.env._pegs[move["source"]]
            dst_xy = self.env._pegs[move["target"]]
            start = self._disk_center(move["disk"])
            end = np.array([move["x"], move["y"], move["z"]])
            delta = end - start
            ax3d.quiver(
                start[0], start[1], start[2] + 0.015,
                delta[0], delta[1], delta[2],
                color="#00A676", linewidth=1.6, arrow_length_ratio=0.12
            )
            ax3d.text(end[0], end[1], end[2] + 0.012,
                      f"{move['disk'].replace('disk_', 'd')}\nz={move['z']:.3f}",
                      ha="center", va="bottom", fontsize=6,
                      color="#006B4A", fontweight="bold")

            ax2d.annotate(
                "",
                xy=(dst_xy["x"], dst_xy["y"]),
                xytext=(src_xy["x"], src_xy["y"]),
                arrowprops=dict(arrowstyle="->", color="#00A676", lw=1.5),
            )
            ax2d.plot(move["x"], move["y"], marker="*", color="#00A676",
                      markersize=9, markeredgecolor="black", markeredgewidth=0.5)

        if self.bounds:
            ax3d.set_xlim(self.bounds["x_min"], self.bounds["x_max"])
            ax3d.set_ylim(self.bounds["y_min"], self.bounds["y_max"])
            ax2d.set_xlim(self.bounds["x_min"], self.bounds["x_max"])
            ax2d.set_ylim(self.bounds["y_min"], self.bounds["y_max"])
        ax3d.set_zlim(0.0, max(0.14, z_top))
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z / stack height (m)")
        ax3d.set_title("3D Hanoi ACFM: legal top-disk moves", fontweight="bold")
        ax3d.view_init(elev=25, azim=-55)

        ax2d.set_aspect("equal", adjustable="box")
        ax2d.grid(True, alpha=0.2)
        ax2d.set_xlabel("X (m)")
        ax2d.set_ylabel("Y (m)")
        ax2d.set_title("Top view: peg centers and legal arrows", fontweight="bold")

        fig.suptitle(
            "Tower of Hanoi ACFM: discrete 3D peg feasibility",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def render_3d_volume(self, save_path=None, threshold=0.0):
        """
        True 3D scatter visualization of the feasibility map.
        Creates a two-panel figure:
          Left: 3D scatter plot (x, y, z) colored by feasibility score.
                Existing objects drawn as 3D AABB boxes.
                Top-3 best positions shown as target-sized 3D boxes.
          Right: 2D projection (max over z) with footprint rectangles.
        Points below `threshold` are hidden for clarity.  By default every grid
        point is shown, with color/size carrying the ACFM value.
        """
        if self.grid3d is None:
            return self.render_heatmap(save_path=save_path)

        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        def _box_faces(cx, cy, cz, hx, hy, hz):
            """Return 6 quad faces of an axis-aligned box."""
            x0, x1 = cx - hx, cx + hx
            y0, y1 = cy - hy, cy + hy
            z0, z1 = cz - hz, cz + hz
            return [
                [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bottom
                [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # top
                [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # front
                [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # back
                [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],  # left
                [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],  # right
            ]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'feasibility',
            ['#C0392B', '#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#1E8449'],
            N=256,
        )

        fig = plt.figure(figsize=(18, 8), dpi=150)

        # ── Left panel: 3D scatter ──
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')

        xs, ys, zs, vals = [], [], [], []
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    s = float(self.grid3d[j, i, k])
                    if s >= threshold:
                        xs.append(float(x))
                        ys.append(float(y))
                        zs.append(float(z))
                        vals.append(s)

        vals_arr = np.array(vals) if vals else np.array([0.5])
        sizes = 8 + 80 * vals_arr

        sc = ax3d.scatter(
            xs, ys, zs, c=vals_arr, cmap=cmap, vmin=0, vmax=1,
            s=sizes, alpha=0.45, edgecolors='none',
        )

        # Existing objects: draw as semi-transparent 3D AABB boxes
        for fp in self.existing_footprints:
            faces = _box_faces(fp['cx'], fp['cy'], fp['cz'],
                               fp['hx'], fp['hy'], fp['hz'])
            poly = Poly3DCollection(faces, alpha=0.35,
                                    facecolor='#888888', edgecolor='white',
                                    linewidth=0.5)
            ax3d.add_collection3d(poly)
            ax3d.text(fp['cx'], fp['cy'], fp['cz'] + fp['hz'] + 0.005,
                      fp['name'], ha='center', va='bottom',
                      fontsize=7, color='white', fontweight='bold')

        # Top-3 best 3D positions: target-sized translucent boxes
        best3d = self.get_best_positions_3d(3)
        thx = self.target_dims[0] / 2.0
        thy = self.target_dims[1] / 2.0
        thz = self.target_dims[2] / 2.0
        for bp in best3d:
            faces = _box_faces(bp['x'], bp['y'], bp['z'], thx, thy, thz)
            poly = Poly3DCollection(faces, alpha=0.45,
                                    facecolor='white', edgecolor='black',
                                    linewidth=1.0)
            ax3d.add_collection3d(poly)
            ax3d.text(bp['x'], bp['y'], bp['z'] + thz + 0.005,
                      f"s={bp['score']:.2f}", ha='center', va='bottom',
                      fontsize=6.5, color='white', fontweight='bold')

        w_cm = round(self.target_dims[0] * 100, 1)
        l_cm = round(self.target_dims[1] * 100, 1)
        h_cm = round(self.target_dims[2] * 100, 1)
        ax3d.set_xlabel('X (m)', fontsize=10)
        ax3d.set_ylabel('Y (m)', fontsize=10)
        ax3d.set_zlabel('Z (m)', fontsize=10)
        ax3d.set_title(
            f'3D ACFM point cloud/grid: "{self.target_name}"\n'
            f'[{w_cm}\u00d7{l_cm}\u00d7{h_cm} cm  (W\u00d7L\u00d7H)]',
            fontsize=11, fontweight='bold',
        )
        ax3d.set_xlim(self.bounds['x_min'], self.bounds['x_max'])
        ax3d.set_ylim(self.bounds['y_min'], self.bounds['y_max'])
        ax3d.set_zlim(0, self.max_z)
        ax3d.view_init(elev=25, azim=-50)

        cbar3d = fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1)
        cbar3d.set_label('Feasibility')
        cbar3d.set_ticks([0, 0.3, 0.7, 1.0])
        cbar3d.set_ticklabels(['Danger', 'Risky', 'Safe', 'Optimal'])

        # ── Right panel: 2D projection with footprint rectangles ──
        ax2d = fig.add_subplot(1, 2, 2)
        extent = [
            self.bounds['x_min'], self.bounds['x_max'],
            self.bounds['y_min'], self.bounds['y_max'],
        ]
        im = ax2d.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            extent=extent, origin='lower', aspect='equal',
            interpolation='nearest',
        )
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.set_title('2D Projection (max over z)', fontsize=12, fontweight='bold')

        # Existing objects: footprint rectangles
        for fp in self.existing_footprints:
            self._draw_rect(
                ax2d, fp['cx'], fp['cy'], fp['hx'], fp['hy'], yaw=fp['yaw'],
                facecolor='#444444', edgecolor='white',
                linewidth=1.2, alpha=0.6, zorder=3,
            )
            ax2d.text(fp['cx'], fp['cy'], fp['name'],
                      ha='center', va='center', fontsize=5,
                      color='white', fontweight='bold', zorder=4, clip_on=True)

        # Target object: dashed rectangles at top-3 positions
        for bp in best3d:
            self._draw_rect(
                ax2d, bp['x'], bp['y'], thx, thy, yaw=0.0,
                facecolor='none', edgecolor='white',
                linewidth=1.5, linestyle='--', zorder=5,
            )
            ax2d.plot(bp['x'], bp['y'], 'w+', markersize=7,
                      markeredgewidth=1.2, zorder=6)
            ax2d.text(
                bp['x'], bp['y'] + thy + 0.008,
                f"z={bp['z']:.2f}  s={bp['score']:.2f}",
                ha='center', va='bottom', fontsize=5.5,
                color='white', fontweight='bold', zorder=6, clip_on=True,
            )

        ax2d.grid(True, alpha=0.2, color='white', linestyle='--')
        cbar2d = fig.colorbar(im, ax=ax2d, shrink=0.8)
        cbar2d.set_label('Feasibility')
        cbar2d.set_ticks([0, 0.3, 0.7, 1.0])
        cbar2d.set_ticklabels(['Danger', 'Risky', 'Safe', 'Optimal'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64


# HanoiACFM is defined between ACFM3D's slice renderer and volume renderer above.
# The following method belongs to ACFM3D, so bind it back explicitly, then give
# HanoiACFM its own discrete peg/stack visualizer.
_ACFM3D_RENDER_3D_VOLUME = HanoiACFM.render_3d_volume
ACFM3D.render_3d_volume = _ACFM3D_RENDER_3D_VOLUME


def _render_hanoi_3d_volume(self, save_path=None):
    if self.env is None:
        return None

    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def _box_faces(cx, cy, cz, hx, hy, hz):
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = cz - hz, cz + hz
        return [
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
        ]

    def _body_color(disk_name):
        try:
            visual = p.getVisualShapeData(self.env.objects[disk_name])
            if visual:
                rgba = tuple(visual[0][7])
                alpha = min(max(rgba[3], 0.35), 0.95)
                return rgba[:3] + (alpha,)
        except Exception:
            pass
        idx = max(0, self._rank(disk_name))
        palette = plt.cm.viridis(np.linspace(0.2, 0.85, max(2, len(self.disk_order))))
        return palette[idx % len(palette)]

    fig = plt.figure(figsize=(16, 7), dpi=150)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hanoi_feasibility",
        ["#7F1D1D", "#C0392B", "#F39C12", "#F1C40F", "#27AE60", "#145A32"],
        N=256,
    )
    if self.grid3d is not None:
        xs, ys, zs, vals = [], [], [], []
        for i, x in enumerate(self.x_range):
            for j, y in enumerate(self.y_range):
                for k, z in enumerate(self.z_range):
                    score = float(self.grid3d[j, i, k])
                    xs.append(float(x))
                    ys.append(float(y))
                    zs.append(float(z))
                    vals.append(score)
        vals_arr = np.array(vals)
        sc = ax3d.scatter(
            xs, ys, zs,
            c=vals_arr,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=6 + 80 * vals_arr,
            alpha=0.42,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax3d, shrink=0.62, pad=0.08)
        cbar.set_label("ACFM feasibility")
        cbar.set_ticks([0, 0.3, 0.7, 1.0])
        cbar.set_ticklabels(["Illegal", "Risky", "Safe", "Best"])

    z_limit = 0.14
    for peg in self.peg_order:
        for disk_name in self.peg_state.get(peg, []):
            if disk_name in self.env.objects:
                aabb = p.getAABB(self.env.objects[disk_name])
                z_limit = max(z_limit, aabb[1][2] + 0.04)
    for move in self.legal_moves:
        dims = self.disk_dims.get(move["disk"], [0.05, 0.05, 0.02])
        z_limit = max(z_limit, move["z"] + dims[2] / 2.0 + 0.04)

    if self.grid is not None and self.bounds:
        ax2d.imshow(
            self.grid,
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[
                self.bounds["x_min"], self.bounds["x_max"],
                self.bounds["y_min"], self.bounds["y_max"],
            ],
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
            alpha=0.58,
        )

    for peg in self.peg_order:
        peg_cfg = self.env._pegs[peg]
        px, py = float(peg_cfg["x"]), float(peg_cfg["y"])
        ax3d.plot([px, px], [py, py], [0.0, z_limit], color="#222222", linewidth=2.0)
        ax3d.text(px, py, z_limit + 0.004, f"Peg {peg}", ha="center",
                  va="bottom", fontsize=8, fontweight="bold")
        ax2d.add_patch(
            Circle((px, py), 0.105, facecolor="#E8F4FD", edgecolor="#2C3E50",
                   linewidth=1.0, alpha=0.65)
        )
        ax2d.text(px, py - 0.13, f"Peg {peg}", ha="center", va="top",
                  fontsize=8, fontweight="bold")

        for disk_name in self.peg_state.get(peg, []):
            if disk_name not in self.env.objects:
                continue
            aabb = p.getAABB(self.env.objects[disk_name])
            cx = (aabb[0][0] + aabb[1][0]) / 2.0
            cy = (aabb[0][1] + aabb[1][1]) / 2.0
            cz = (aabb[0][2] + aabb[1][2]) / 2.0
            hx = (aabb[1][0] - aabb[0][0]) / 2.0
            hy = (aabb[1][1] - aabb[0][1]) / 2.0
            hz = (aabb[1][2] - aabb[0][2]) / 2.0
            color = _body_color(disk_name)
            ax3d.add_collection3d(
                Poly3DCollection(
                    _box_faces(cx, cy, cz, hx, hy, hz),
                    facecolor=color, edgecolor="black",
                    linewidth=0.4, alpha=color[3],
                )
            )
            ax3d.text(cx, cy, cz + hz + 0.002, disk_name.replace("disk_", "d"),
                      ha="center", va="bottom", fontsize=6)

            ax2d.add_patch(
                Circle((cx, cy), max(hx, hy), facecolor=color,
                       edgecolor="black", linewidth=0.6, alpha=0.75)
            )
            if disk_name == self.peg_state.get(peg, [])[-1]:
                ax2d.text(cx, cy, disk_name.replace("disk_", "d"),
                          ha="center", va="center", fontsize=6,
                          fontweight="bold", color="black")

    for move in self.legal_moves:
        src_xy = self.env._pegs[move["source"]]
        dst_xy = self.env._pegs[move["target"]]
        start = self._disk_center(move["disk"])
        end = np.array([move["x"], move["y"], move["z"]])
        delta = end - start
        ax3d.quiver(
            start[0], start[1], start[2] + 0.015,
            delta[0], delta[1], delta[2],
            color="#00A676", linewidth=1.6, arrow_length_ratio=0.12,
        )
        ax3d.text(end[0], end[1], end[2] + 0.012,
                  f"{move['disk'].replace('disk_', 'd')}\nz={move['z']:.3f}",
                  ha="center", va="bottom", fontsize=6,
                  color="#006B4A", fontweight="bold")

        ax2d.annotate(
            "",
            xy=(dst_xy["x"], dst_xy["y"]),
            xytext=(src_xy["x"], src_xy["y"]),
            arrowprops=dict(arrowstyle="->", color="#00A676", lw=1.5),
        )
        ax2d.plot(move["x"], move["y"], marker="*", color="#00A676",
                  markersize=9, markeredgecolor="black", markeredgewidth=0.5)

    if self.bounds:
        ax3d.set_xlim(self.bounds["x_min"], self.bounds["x_max"])
        ax3d.set_ylim(self.bounds["y_min"], self.bounds["y_max"])
        ax2d.set_xlim(self.bounds["x_min"], self.bounds["x_max"])
        ax2d.set_ylim(self.bounds["y_min"], self.bounds["y_max"])
    ax3d.set_zlim(0.0, z_limit)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z / stack height (m)")
    ax3d.set_title("3D Hanoi ACFM cloud/grid: legal top-disk moves", fontweight="bold")
    ax3d.view_init(elev=25, azim=-55)

    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, alpha=0.2)
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_title("Top view: peg centers and legal arrows", fontweight="bold")

    fig.suptitle(
        "Tower of Hanoi ACFM: discrete 3D peg feasibility",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


HanoiACFM.render_3d_volume = _render_hanoi_3d_volume
