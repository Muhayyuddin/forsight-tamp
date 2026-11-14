import json
from pathlib import Path

from envs.pack_compact_env import PackCompactEnv
from main_foresight import ForeSightTAMP
from utils.tamp_util import text_to_actions


def test_pick_and_place_paths_export_joint_and_cartesian_phases(tmp_path):
    scenario_path = (
        Path(__file__).parents[1]
        / "tests"
        / "fixtures"
        / "PKG2_tomato_cans_6_box.json"
    )
    with scenario_path.open() as stream:
        scenario = json.load(stream)

    env = PackCompactEnv()
    try:
        env.reset(
            basket=scenario["basket"],
            boxes=scenario["boxes"],
            task_type="packing",
            use_gui=False,
        )
        foresight = ForeSightTAMP(
            env=env,
            use_gui=False,
            save_dir=tmp_path,
        )
        plan = text_to_actions(
            [
                "pick(['tomato_soup_can_1'], {})",
                (
                    "place(['tomato_soup_can_1'], "
                    "{'x': 0.357, 'y': -0.065, 'theta': 0.0})"
                ),
            ],
            env.primitive_actions,
        )

        joint_records = []
        cartesian_records = []
        for action_index, action in enumerate(plan):
            _, feedback = env.step(action, play_traj=False)
            assert feedback.action_success
            joint_record, cartesian_record = (
                foresight._capture_executed_action_paths(
                    action=action,
                    iteration=0,
                    action_index=action_index,
                )
            )
            joint_records.append(joint_record)
            cartesian_records.append(cartesian_record)

        foresight._write_executed_path_files(
            joint_records, cartesian_records)

        with (tmp_path / "executed_joint_space_paths.json").open() as stream:
            joint_data = json.load(stream)
        with (tmp_path / "executed_cartesian_paths.json").open() as stream:
            cartesian_data = json.load(stream)
        with (
            tmp_path / "executed_interpolated_joint_space_paths.json"
        ).open() as stream:
            interpolated_joint_data = json.load(stream)
        with (
            tmp_path / "executed_interpolated_cartesian_paths.json"
        ).open() as stream:
            interpolated_cartesian_data = json.load(stream)

        assert len(joint_data["actions"]) == 2
        assert len(cartesian_data["actions"]) == 2
        assert interpolated_joint_data == joint_data
        assert interpolated_cartesian_data == cartesian_data
        assert joint_data["format_version"] == 2
        assert cartesian_data["format_version"] == 2
        assert joint_data["path_kind"] == (
            "interpolated_execution_trajectory"
        )
        assert joint_data["interpolation"]["method"] == (
            "quintic_smoothstep"
        )
        assert joint_data["time_step_seconds"] == cartesian_data[
            "time_step_seconds"
        ]
        assert joint_data["timing_profile"] == (
            "quintic_stop_at_polyline_corners"
        )
        assert [
            segment["phase"]
            for action in joint_data["actions"]
            for segment in action["segments"]
        ] == [
            "pick_pregrasp",
            "pick_cartesian_descent",
            "post_grasp_cartesian_lift",
            "place_transfer",
            "post_place_lift",
        ]

        for joint_action, cartesian_action in zip(
                joint_data["actions"], cartesian_data["actions"]):
            for joint_segment, cartesian_segment in zip(
                    joint_action["segments"],
                    cartesian_action["segments"]):
                assert (
                    joint_segment["waypoint_count"]
                    == cartesian_segment["waypoint_count"]
                )
                assert len(
                    joint_segment["waypoints"][0]
                ) == len(joint_data["joint_names"])
                assert len(
                    cartesian_segment["waypoints"][0]["position_xyz"]
                ) == 3
                assert len(
                    cartesian_segment["waypoints"][0][
                        "orientation_xyzw"]
                ) == 4

        finger_indices = [
            index
            for index, name in enumerate(joint_data["joint_names"])
            if "finger_joint" in name
        ]
        for finger_index in finger_indices:
            values = [
                waypoint[finger_index]
                for action in joint_data["actions"]
                for segment in action["segments"]
                for waypoint in segment["waypoints"]
            ]
            assert max(values) - min(values) < 1e-12
    finally:
        env.destroy()
