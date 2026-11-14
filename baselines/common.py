"""Shared, method-neutral utilities for adapted baseline evaluations.

These helpers expose the same primitive actions and task models to every baseline.
They deliberately do not use ForeSight's ACFM, rehearsal, or cascade modules.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

import numpy as np

from utils.tamp_util import Action, text_to_actions


SUPPORTED_ENV_TYPES = {
    "packing", "stacking", "shelf", "push_stacking", "hanoi",
}


def infer_env_type(env) -> str:
    name = type(env).__name__
    return {
        "PackCompactEnv": "packing",
        "StackingEnv": "stacking",
        "ShelfEnv": "shelf",
        "PushStackingEnv": "push_stacking",
        "TowerOfHanoiEnv": "hanoi",
    }.get(name, "unknown")


def primitive_schema(primitive_actions) -> dict[str, Any]:
    return {
        name: {
            "object_arity": primitive.obj_arity,
            "parameters": {
                pname: [param.lower_limit, param.upper_limit]
                for pname, param in primitive.parameters.items()
            },
        }
        for name, primitive in primitive_actions.items()
    }


def task_objects(task_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task_config.get("boxes", task_config.get("disks", []))
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [dict(item) for item in raw if isinstance(item, dict) and item.get("name")]


def held_object(env) -> str | None:
    """Resolve the attachment child body, rather than guessing from object height."""
    attachments = getattr(getattr(env, "robot", None), "attachments_robot", [])
    if not attachments:
        return None
    child = getattr(attachments[0], "child", None)
    for name, body in getattr(env, "objects", {}).items():
        if body == child:
            return name
    return None


def safe_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _packing_targets(task_config: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    """Generate a neutral regular grid; this is not an ACFM or optimized packer."""
    basket = task_config["basket"]
    objects = task_objects(task_config)
    if not objects:
        return []
    max_w = max(float(o.get("w", 0.05)) for o in objects)
    max_l = max(float(o.get("l", o.get("w", 0.05))) for o in objects)
    xmin = basket["x"] - basket["w"] / 2 + max_w / 2 + 0.008
    xmax = basket["x"] + basket["w"] / 2 - max_w / 2 - 0.008
    ymin = basket["y"] - basket["l"] / 2 + max_l / 2 + 0.008
    ymax = basket["y"] + basket["l"] / 2 - max_l / 2 - 0.008
    cols = max(1, int((xmax - xmin) / max(max_w + 0.008, 1e-3)) + 1)
    rows = max(1, math.ceil(len(objects) / cols))
    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    slots = [(float(x), float(y)) for y in ys for x in xs]
    # A deterministic size-first order is a standard geometric heuristic, not a
    # task-instance solution.
    objects = sorted(
        objects,
        key=lambda o: float(o.get("w", .05)) * float(o.get("l", o.get("w", .05))),
        reverse=True,
    )
    return [
        (obj["name"], {"x": x, "y": y, "theta": float(obj.get("init_theta", 0.0))})
        for obj, (x, y) in zip(objects, slots)
    ]


def grounded_reference_plan(env, task_config: dict[str, Any]) -> list[Action]:
    """Return a domain planner's grounded plan without ForeSight-only guidance."""
    env_type = infer_env_type(env)
    actions: list[str] = []

    if env_type == "hanoi":
        actions = _solve_hanoi_symbolically(env)
    elif env_type == "push_stacking":
        actions = list(env.get_symbolic_plan())
    elif env_type == "packing":
        for name, params in _packing_targets(task_config):
            actions.extend([
                f"pick([{name!r}], {{}})",
                f"place([{name!r}], {params!r})",
            ])
    elif env_type == "stacking":
        zone = task_config.get("stacking_zone") or {"x": 0.55, "y": 0.0}
        for name in task_config.get("stacking_order", []):
            params = {
                "x": float(zone["x"]), "y": float(zone["y"]),
                "z": 0.02, "theta": 0.0,
            }
            actions.extend([f"pick([{name!r}], {{}})", f"place([{name!r}], {params!r})"])
    elif env_type == "shelf":
        cfg = task_config["shelf_config"]
        specs = {o["name"]: o for o in task_objects(task_config)}
        for level_idx, level in enumerate(cfg["levels"]):
            names = list(level.get("objects", []))
            half_w = cfg["width"] / 2
            margin = max([float(specs.get(n, {}).get("w", .05)) / 2 for n in names] or [.025])
            xs = np.linspace(cfg["x"] - half_w + margin, cfg["x"] + half_w - margin, max(1, len(names)))
            for name, x in zip(names, xs):
                z = env._minimum_center_z_for_level(name, level_idx)
                params = {
                    "x": float(x),
                    "y": float(cfg["y"] - cfg["depth"] * 0.25),
                    "z": float(z),
                    "theta": 0.0,
                }
                actions.extend([f"pick([{name!r}], {{}})", f"place([{name!r}], {params!r})"])
    else:
        raise ValueError(f"Unsupported baseline environment: {env_type}")

    return text_to_actions(actions, env.primitive_actions)


def _solve_hanoi_symbolically(env) -> list[str]:
    """Breadth-first symbolic planning for arbitrary disk counts and peg states."""
    from collections import deque

    disks = list(env._disks_ordered)  # largest -> smallest
    peg_names = tuple(env._pegs)
    initial_map = env._get_peg_state()
    initial = tuple(tuple(initial_map[p]) for p in peg_names)
    goal = tuple(tuple(disks) if p == "C" else tuple() for p in peg_names)
    queue = deque([initial])
    parent = {initial: None}
    parent_move = {}

    while queue:
        state = queue.popleft()
        if state == goal:
            break
        for src_i, src in enumerate(state):
            if not src:
                continue
            disk = src[-1]
            disk_rank = disks.index(disk)
            for dst_i, dst in enumerate(state):
                if src_i == dst_i:
                    continue
                # A destination top with a greater rank is physically smaller.
                if dst and disks.index(dst[-1]) > disk_rank:
                    continue
                next_state = [list(stack) for stack in state]
                next_state[src_i].pop()
                next_state[dst_i].append(disk)
                nxt = tuple(tuple(stack) for stack in next_state)
                if nxt in parent:
                    continue
                parent[nxt] = state
                parent_move[nxt] = (disk, peg_names[dst_i])
                queue.append(nxt)
    if goal not in parent:
        raise ValueError("No legal symbolic Hanoi plan exists")

    moves = []
    cursor = goal
    while parent[cursor] is not None:
        moves.append(parent_move[cursor])
        cursor = parent[cursor]
    moves.reverse()
    actions = []
    simulated = {p: list(initial[i]) for i, p in enumerate(peg_names)}
    for disk, dst in moves:
        for stack in simulated.values():
            if disk in stack:
                stack.remove(disk)
                break
        z = env._next_place_z(dst, simulated, disk)
        simulated[dst].append(disk)
        peg = env._pegs[dst]
        actions.extend([
            f"pick([{disk!r}], {{}})",
            (
                f"place([{disk!r}], {{'x': {float(peg['x'])}, "
                f"'y': {float(peg['y'])}, 'z': {float(z)}, 'theta': 0.0}})"
            ),
        ])
    return actions


def grounded_skill_library(env, task_config: dict[str, Any]) -> list[Action]:
    """Enumerate predefined skills without exposing a solution sequence."""
    if infer_env_type(env) != "hanoi":
        return grounded_reference_plan(env, task_config)
    raw = []
    for disk in env._disks_ordered:
        raw.append(f"pick([{disk!r}], {{}})")
        for peg in env._pegs.values():
            raw.append(
                f"place([{disk!r}], {{'x': {float(peg['x'])}, "
                f"'y': {float(peg['y'])}, 'z': 0.02, 'theta': 0.0}})"
            )
    return text_to_actions(raw, env.primitive_actions)


def action_key(action: Action) -> str:
    return str(action)


def action_is_admissible(env, action: Action) -> bool:
    """Symbolic/precondition affordance shared by baselines."""
    primitive = action.primitive.name
    obj = action.obj_args[0] if action.obj_args else None
    held = held_object(env)
    if primitive == "pick":
        if held is not None or obj not in env.objects:
            return False
        if hasattr(env, "_graspable") and not env._graspable.get(obj, True):
            return False
        if infer_env_type(env) == "hanoi":
            state = env._get_peg_state()
            peg = env._find_disk_peg(obj, state)
            return bool(peg and state[peg] and state[peg][-1] == obj)
        return True
    if primitive == "place":
        if held != obj:
            return False
        if infer_env_type(env) == "hanoi":
            target = env._peg_from_xy(
                action.param_args.get("x", float("inf")),
                action.param_args.get("y", float("inf")),
            )
            if target is None:
                return False
            state = env._remove_disk_from_peg_state(env._get_peg_state(), obj)
            top = state[target][-1] if state[target] else None
            return top is None or env._is_smaller_disk(obj, top)
        return True
    if primitive == "push":
        return held is None and bool(getattr(env, "_is_pushable", lambda _o: False)(obj))
    return False


def remaining_admissible_skills(env, plan: Iterable[Action], executed: set[str]) -> list[Action]:
    return [
        action for action in plan
        if action_key(action) not in executed and action_is_admissible(env, action)
    ]


def execute_action(env, action: Action, play_traj: bool) -> tuple[Any, Any]:
    return env.step(action, play_traj=play_traj)
