from typing import List, Dict, Any
from dataclasses import dataclass, field

import ast
import logging

logger = logging.getLogger(__name__)


@dataclass
class TAMPFeedback:
    motion_planner_feedback: str = ""
    task_process_feedback: str = ""
    action_success: bool = False
    goal_achieved: bool = False


@dataclass
class Parameter:
    name: str
    lower_limit: float
    upper_limit: float


class CachedTrajectory(list):
    """Joint waypoints plus the validated intra-action phase boundaries.

    A pick trajectory can contain motion both before and after attachment, and
    a place trajectory can contain motion both before and after release.  A
    plain list cannot represent those state-changing boundaries.  Keeping the
    primary waypoints list-compatible preserves existing planners while the
    ``motion_trace`` attribute lets execution replay attachment/detachment at
    exactly the phases exercised during shadow rollout.
    """

    def __init__(self, waypoints=(), motion_trace=None):
        super().__init__(waypoints)
        self.motion_trace = list(motion_trace or [])


@dataclass
class Action:
    primitive: "PrimitiveAction"
    obj_args: list = field(default_factory=list)
    param_args: dict = field(default_factory=dict)
    traj: list = field(default_factory=list)
    motion_trace: list = field(default_factory=list)

    def __str__(self):
        return f"{self.primitive.name}({self.obj_args}, {self.param_args})"


@dataclass
class PrimitiveAction:
    name: str
    obj_arity: int = 0
    parameters: dict = field(default_factory=dict)

    def __call__(
        self,
        obj_args: List[str],
        param_args: Dict[str, float] | None = None,
        traj: List[Any] | None = None,
    ):
        # Copy caller-owned collections before clamping parameters so that
        # actions never share mutable dictionaries or trajectories.
        obj_args = list(obj_args)
        param_args = dict(param_args or {})
        traj = [] if traj is None else traj
        assert len(obj_args) == self.obj_arity, f"Expected {self.obj_arity} object arguments!"
        # Allow a subset of parameters (e.g. z is optional for some environments)
        unknown = [p for p in param_args if p not in self.parameters]
        assert not unknown, f"Unknown param arguments: {unknown}!"

        for param in param_args.keys():
            if param not in self.parameters:
                param_args.pop(param)
                logger.warn(f"Unknown param argument {param}!")
            else:
                # check if param is within limits
                lower_limit, upper_limit = (
                    self.parameters[param].lower_limit,
                    self.parameters[param].upper_limit,
                )
                if param_args[param] < lower_limit:
                    param_args[param] = lower_limit
                    logger.warn(
                        f"Param argument {param} is below lower limit {lower_limit}! Setting it to {lower_limit}."
                    )
                elif param_args[param] > upper_limit:
                    param_args[param] = upper_limit
                    logger.warn(
                        f"Param argument {param} is above upper limit {upper_limit}! Setting it to {upper_limit}."
                    )

        return Action(self, obj_args, param_args, traj)


def create_action_from_raw(raw_action: str, primitive_actions: Dict[str, PrimitiveAction]):
    """Parse one symbolic action without executing model-provided text.

    Only the expected ``primitive([objects], {parameters})`` expression is
    parsed, and both arguments are restricted to Python literals, so LLM output
    is never evaluated as code.
    """
    try:
        expression = ast.parse(raw_action.strip(), mode="eval").body
    except (SyntaxError, TypeError) as exc:
        raise ValueError(f"Malformed action: {raw_action!r}") from exc

    if not isinstance(expression, ast.Call) or not isinstance(
        expression.func, ast.Name
    ):
        raise ValueError(f"Malformed action: {raw_action!r}")
    if expression.keywords or len(expression.args) not in (1, 2):
        raise ValueError(
            "Actions require an object list and an optional parameter dictionary"
        )

    primitive_name = expression.func.id
    if primitive_name not in primitive_actions:
        raise ValueError(f"Unknown primitive action: {primitive_name!r}")
    primitive_action = primitive_actions[primitive_name]

    try:
        obj_args = ast.literal_eval(expression.args[0])
        param_args = (
            ast.literal_eval(expression.args[1])
            if len(expression.args) == 2
            else {}
        )
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Action arguments must be literals: {raw_action!r}") from exc

    if not isinstance(obj_args, (list, tuple)) or not all(
        isinstance(obj, str) for obj in obj_args
    ):
        raise ValueError("Action object arguments must be a list of object names")
    if not isinstance(param_args, dict) or not all(
        isinstance(name, str) and isinstance(value, (int, float))
        and not isinstance(value, bool)
        for name, value in param_args.items()
    ):
        raise ValueError(
            "Action parameters must be a dictionary of numeric literal values"
        )

    return primitive_action(obj_args=obj_args, param_args=param_args)


def text_to_actions(text: List[str], primitive_actions: Dict[str, PrimitiveAction]):
    action_plan = []

    for raw_action in text:
        action = create_action_from_raw(raw_action, primitive_actions)
        action_plan.append(action)

    return action_plan
