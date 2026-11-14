"""Shared conversion from scenario dynamics fields to PyBullet arguments."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCENARIO_TO_PYBULLET = {
    "mass": "mass",
    "lateral_friction": "lateralFriction",
    "rolling_friction": "rollingFriction",
    "spinning_friction": "spinningFriction",
    "linear_damping": "linearDamping",
    "angular_damping": "angularDamping",
    "restitution": "restitution",
}

_NONNEGATIVE_FIELDS = {
    "mass",
    "lateral_friction",
    "rolling_friction",
    "spinning_friction",
    "linear_damping",
    "angular_damping",
}


def apply_configured_dynamics(
    body_id: int,
    config: Mapping[str, Any],
    *,
    pybullet_module=None,
) -> dict[str, float]:
    """Apply only explicitly configured dynamics and return applied values.

    Keeping this conversion in one place makes scenario dynamics fields take
    effect in the packing, shelf, stacking, and Hanoi environments.
    """
    if pybullet_module is None:
        import pybullet as pybullet_module

    kwargs: dict[str, float] = {}
    for scenario_name, pybullet_name in SCENARIO_TO_PYBULLET.items():
        if scenario_name in config:
            raw_value = config[scenario_name]
            if isinstance(raw_value, bool):
                raise ValueError(f"{scenario_name} must be a finite number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{scenario_name} must be a finite number"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{scenario_name} must be a finite number")
            if scenario_name in _NONNEGATIVE_FIELDS and value < 0:
                raise ValueError(f"{scenario_name} must be nonnegative")
            if scenario_name == "restitution" and not 0.0 <= value <= 1.0:
                raise ValueError("restitution must be between 0 and 1")
            kwargs[pybullet_name] = value
    if kwargs:
        pybullet_module.changeDynamics(body_id, -1, **kwargs)
    return kwargs
