from envs.pb_env import PyBulletRobot
from envs.shelf_env import ShelfEnv
from utils.tamp_util import CachedTrajectory


def test_shelf_pick_requests_postgrasp_table_clearance():
    env = ShelfEnv()
    env._objects = {"box": 13, "table": 12}
    env.prepare_obstacles = lambda **_kwargs: {12: "table"}
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
    assert captured["postgrasp_clearance"] == 0.06


def test_direct_side_pick_lift_and_shelf_retreat_are_cacheable():
    pick = CachedTrajectory(
        [(0.0,), (0.1,), (0.2,)],
        motion_trace=[
            {"phase": "pick_approach", "trajectory": [(0.0,), (0.1,)]},
            {
                "phase": "post_grasp_cartesian_lift",
                "trajectory": [(0.1,), (0.2,)],
            },
        ],
    )
    place = CachedTrajectory(
        [(0.2,), (0.3,)],
        motion_trace=[
            {"phase": "place_transfer", "trajectory": [(0.2,), (0.3,)]},
            {
                "phase": "post_place_retreat",
                "trajectory": [(0.3,), (0.4,)],
            },
        ],
    )

    assert [
        phase for phase, _ in PyBulletRobot._validated_cached_segments(
            pick, "pick")
    ] == ["pick_approach", "post_grasp_cartesian_lift"]
    assert [
        phase for phase, _ in PyBulletRobot._validated_cached_segments(
            place, "place")
    ] == ["place_transfer", "post_place_retreat"]
