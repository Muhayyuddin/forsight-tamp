from types import SimpleNamespace

from baselines.common import (
    _packing_targets,
    held_object,
    infer_env_type,
    primitive_schema,
)
from envs.pack_compact_env import PackCompactEnv
from utils.tamp_util import Parameter, PrimitiveAction


def test_environment_type_is_explicit():
    assert infer_env_type(PackCompactEnv.__new__(PackCompactEnv)) == "packing"
    assert infer_env_type(SimpleNamespace()) == "unknown"


def test_held_object_uses_attachment_identity_not_height():
    attachment = SimpleNamespace(child=22)
    env = SimpleNamespace(
        robot=SimpleNamespace(attachments_robot=[attachment]),
        objects={"short_can": 22, "other": 23},
    )
    assert held_object(env) == "short_can"


def test_neutral_packing_grid_is_finite_and_inside_container():
    task = {
        "basket": {"x": 0.6, "y": 0.0, "w": 0.4, "l": 0.2},
        "boxes": {
            str(i): {"name": f"can_{i}", "w": 0.05, "l": 0.05}
            for i in range(6)
        },
    }
    targets = _packing_targets(task)
    assert len(targets) == 6
    assert len({(p["x"], p["y"]) for _, p in targets}) == 6
    assert all(0.4 < p["x"] < 0.8 and -0.1 < p["y"] < 0.1 for _, p in targets)


def test_primitive_schema_preserves_continuous_bounds():
    actions = {
        "place": PrimitiveAction(
            "place",
            obj_arity=1,
            parameters={"x": Parameter("x", 0.0, 1.4)},
        )
    }
    assert primitive_schema(actions)["place"]["parameters"]["x"] == [0.0, 1.4]
