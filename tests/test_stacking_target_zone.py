from envs.stacking_env import StackingEnv


def test_visual_target_zone_is_not_a_motion_obstacle():
    env = StackingEnv.__new__(StackingEnv)
    env._objects = {"target_zone": 11, "table": 12, "box": 13}
    env.prepare_obstacles = lambda **_kwargs: {
        11: "target_zone",
        12: "table",
    }

    assert env._prepare_motion_obstacles("box") == {12: "table"}


def test_stacking_pick_uses_vertical_clearance_before_lateral_transfer():
    env = StackingEnv()
    env._objects = {"box": 13, "table": 12}
    env._box_specs = {"box": {}}
    env._prepare_motion_obstacles = lambda _name: {12: "table"}
    captured = {}

    class Robot:
        def pick(self, *args, **kwargs):
            captured.update(kwargs)
            return True, [(0.0,)], "Success"

    env.robot = Robot()
    action = env.primitive_actions["pick"](["box"], {})

    success, feedback = env.apply_action(action, play_traj=False)

    assert success
    assert feedback == "Success"
    assert captured["pregrasp_clearance"] == 0.08
