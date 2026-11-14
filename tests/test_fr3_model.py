import xml.etree.ElementTree as ET

import pytest

import envs.pb_env as pb_env
from envs.pb_env import (
    FR3_ARM_JOINT_NAMES,
    FRANKA_ROBOT_URDF,
    PyBulletRobot,
)


EXPECTED_FR3_LIMITS = {
    "fr3_joint1": (-2.9007, 2.9007),
    "fr3_joint2": (-1.8361, 1.8361),
    "fr3_joint3": (-2.9007, 2.9007),
    "fr3_joint4": (-3.0770, -0.1169),
    "fr3_joint5": (-2.8763, 2.8763),
    "fr3_joint6": (0.4398, 4.6216),
    "fr3_joint7": (-3.0508, 3.0508),
}


def test_official_fr3_description_is_selected():
    root = ET.parse(FRANKA_ROBOT_URDF).getroot()

    assert root.attrib["name"] == "fr3"
    assert root.find("./link[@name='fr3_hand_tcp']") is not None
    assert root.find("./link[@name='panda_link0']") is None


def test_fr3_arm_limits_match_robot_specification():
    root = ET.parse(FRANKA_ROBOT_URDF).getroot()
    actual = {}
    for joint_name in FR3_ARM_JOINT_NAMES:
        joint = root.find(f"./joint[@name='{joint_name}']")
        assert joint is not None
        limit = joint.find("limit")
        actual[joint_name] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )

    for name, expected in EXPECTED_FR3_LIMITS.items():
        assert actual[name] == pytest.approx(expected)


def test_fr3_collision_meshes_are_vendored():
    root = ET.parse(FRANKA_ROBOT_URDF).getroot()

    # FRANKA_ROBOT_URDF is currently an os.path string. Resolve every arm
    # collision filename relative to the URDF, exactly as PyBullet does.
    from pathlib import Path

    urdf_dir = Path(FRANKA_ROBOT_URDF).parent
    for index in range(8):
        mesh = root.find(
            f"./link[@name='fr3_link{index}']/collision/geometry/mesh"
        )
        assert mesh is not None
        assert (urdf_dir / mesh.attrib["filename"]).resolve().is_file()


def test_tiny_ik_solver_spill_is_projected_inside_safe_planning_limit(
        monkeypatch):
    robot = PyBulletRobot.__new__(PyBulletRobot)
    robot.robot = 7
    robot.ik_joints = [0, 1]
    robot.planning_limits = {0: (-1.0, 1.0), 1: (-2.0, 2.0)}
    monkeypatch.setattr(
        pb_env.p,
        "getJointInfo",
        lambda _robot, joint: (None, f"joint_{joint}".encode("utf-8")),
    )

    projected, feedback = robot._project_small_ik_limit_spill(
        (-1.0004, 0.25), "IK solution")

    assert feedback is None
    assert projected == pytest.approx((-1.0, 0.25))


def test_material_ik_limit_violation_remains_a_hard_failure(monkeypatch):
    robot = PyBulletRobot.__new__(PyBulletRobot)
    robot.robot = 7
    robot.ik_joints = [0]
    robot.planning_limits = {0: (-1.0, 1.0)}
    monkeypatch.setattr(
        pb_env.p,
        "getJointInfo",
        lambda _robot, _joint: (None, b"joint_0"),
    )

    projected, feedback = robot._project_small_ik_limit_spill(
        (-1.01,), "IK solution")

    assert projected is None
    assert feedback == (
        "IK solution violates joint_0 position limit: "
        "-1.010000 not in [-1.000000, 1.000000]"
    )
