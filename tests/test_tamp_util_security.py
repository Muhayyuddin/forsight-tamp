import pytest

from utils.tamp_util import Parameter, PrimitiveAction, create_action_from_raw


def _primitives():
    return {
        "place": PrimitiveAction(
            "place",
            obj_arity=1,
            parameters={"x": Parameter("x", -1.0, 1.0)},
        )
    }


def test_action_parser_accepts_only_literal_arguments():
    action = create_action_from_raw(
        "place(['box'], {'x': 0.25})",
        _primitives(),
    )
    assert action.obj_args == ["box"]
    assert action.param_args == {"x": 0.25}

    with pytest.raises(ValueError, match="must be literals"):
        create_action_from_raw(
            "place([__import__('os').getcwd()], {'x': 0.25})",
            _primitives(),
        )


def test_primitive_actions_do_not_share_default_collections():
    primitive = PrimitiveAction("pick", obj_arity=1)
    first = primitive(["box_a"])
    second = primitive(["box_b"])

    first.param_args["mutated"] = 1
    first.traj.append((0.0,))

    assert second.param_args == {}
    assert second.traj == []
