"""Replaying a finished run in the window must not change what the run does.

The capture hook sits inside the simulator's animation loop, which is also the
path every benchmark trial executes. It is therefore only safe if it is inert
unless a caller installs a sink, and the benchmark never installs one -- the
recorded study has to stay reproducible against this tree.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import replay_capture  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_sink():
    replay_capture.set_sink(None)
    yield
    replay_capture.set_sink(None)


def test_no_sink_is_installed_by_default():
    assert replay_capture.active() is False


def test_frame_is_a_no_op_without_a_sink():
    replay_capture.frame()  # must not raise


def test_frame_reaches_an_installed_sink():
    calls = []
    replay_capture.set_sink(lambda: calls.append(1))
    replay_capture.frame()
    replay_capture.frame()
    assert len(calls) == 2


def test_clearing_the_sink_stops_capture():
    calls = []
    replay_capture.set_sink(lambda: calls.append(1))
    replay_capture.frame()
    replay_capture.set_sink(None)
    replay_capture.frame()
    assert len(calls) == 1
    assert replay_capture.active() is False


def test_set_sink_returns_the_previous_one():
    first = (lambda: None)
    assert replay_capture.set_sink(first) is None
    assert replay_capture.set_sink(None) is first


def test_benchmark_runner_installs_no_sink():
    """generate_results.py is the measured path; it must never capture."""
    source = (ROOT / "generate_results.py").read_text()
    assert "set_sink" not in source
    assert "replay_capture" not in source


def _world():
    pybullet = pytest.importorskip("pybullet")
    client = pybullet.connect(pybullet.DIRECT)
    return pybullet, client


def test_track_captures_and_restores_body_state():
    pybullet, client = _world()
    try:
        import run_foresight as rf

        shape = pybullet.createCollisionShape(pybullet.GEOM_BOX,
                                              halfExtents=[0.1, 0.1, 0.1])
        body = pybullet.createMultiBody(1.0, shape,
                                        basePosition=[0.0, 0.0, 1.0])
        track = rf.ReplayTrack()
        track.capture()
        assert len(track) == 1

        pybullet.resetBasePositionAndOrientation(body, [5.0, 5.0, 5.0],
                                                 [0, 0, 0, 1])
        moved = pybullet.getBasePositionAndOrientation(body)[0]
        assert moved[0] == pytest.approx(5.0)

        track._show(track.frames[0])
        restored = pybullet.getBasePositionAndOrientation(body)[0]
        assert restored[2] == pytest.approx(1.0)
        assert restored[0] == pytest.approx(0.0)
    finally:
        pybullet.disconnect(client)


def test_track_stride_keeps_every_nth_frame():
    pybullet, client = _world()
    try:
        import run_foresight as rf

        track = rf.ReplayTrack(stride=3)
        for _ in range(7):
            track.capture()
        assert len(track) == 3
    finally:
        pybullet.disconnect(client)


def test_hook_fires_once_per_animated_frame():
    """The sink must see every frame the window draws, not every action."""
    from unittest import mock

    from envs.pb_env import PyBulletRobot

    seen = []
    replay_capture.set_sink(lambda: seen.append(1))
    with mock.patch("envs.pb_env.set_joint_positions"), \
            mock.patch("envs.pb_env.wait_for_duration"):
        PyBulletRobot.simulate_traj(
            None, robot=1, ik_joints=[0], attachments_robot=[], attached=None,
            tool_attach_link=None, ee_link_from_tcp=None,
            traj=[(0.0,), (0.1,), (0.2,), (0.3,)], play_traj=True)
    assert len(seen) == 4


def test_hook_is_silent_when_the_run_is_not_animated():
    """Headless trials skip the animation loop, so nothing should be captured."""
    from unittest import mock

    from envs.pb_env import PyBulletRobot

    seen = []
    replay_capture.set_sink(lambda: seen.append(1))
    with mock.patch("envs.pb_env.set_joint_positions"), \
            mock.patch("envs.pb_env.wait_for_duration"):
        PyBulletRobot.simulate_traj(
            None, robot=1, ik_joints=[0], attachments_robot=[], attached=None,
            tool_attach_link=None, ee_link_from_tcp=None,
            traj=[(0.0,), (0.1,), (0.2,)], play_traj=False)
    assert seen == []


def test_camera_pose_is_printed_as_the_call_that_restores_it():
    """A viewpoint worth keeping is only useful if it can be pasted back."""
    import run_foresight as rf

    line = rf.format_camera_pose((50.0, -35.0, 1.2, (0.4, 0.0, 0.25)))
    assert line.startswith("p.resetDebugVisualizerCamera(")
    assert "cameraDistance=1.20" in line
    assert "cameraYaw=50.0" in line
    assert "cameraPitch=-35.0" in line
    assert "cameraTargetPosition=[0.400, 0.000, 0.250]" in line


def test_camera_pose_reads_the_four_placement_fields():
    pybullet, client = _world()
    try:
        import run_foresight as rf

        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=30.0, cameraPitch=-20.0,
            cameraTargetPosition=[0.1, 0.2, 0.3])
        yaw, pitch, distance, target = rf.camera_pose()
        assert isinstance(yaw, float) and isinstance(pitch, float)
        assert isinstance(distance, float)
        assert len(target) == 3
    finally:
        pybullet.disconnect(client)


def _watching(module, poses):
    """Patch camera_pose with a sequence and count the lines logged."""
    from unittest import mock

    values = list(poses)

    def next_pose():
        return values.pop(0) if len(values) > 1 else values[0]

    return (mock.patch.object(module, "camera_pose", side_effect=next_pose),
            mock.patch.object(module.logger, "info"))


def test_camera_watch_reports_only_when_the_view_moves():
    """A still camera must not repeat itself on every frame."""
    import run_foresight as rf

    poses = [(50.0, -35.0, 1.2, (0.0, 0.0, 0.0)),
             (50.0, -35.0, 1.2, (0.0, 0.0, 0.0)),
             (90.0, -35.0, 1.2, (0.0, 0.0, 0.0))]
    pose_patch, log_patch = _watching(rf, poses)
    watch = rf.CameraWatch(min_interval=0.0)
    with pose_patch, log_patch as logged:
        watch.poll()
        watch.poll()
        watch.poll()
    assert logged.call_count == 2


def test_camera_watch_is_rate_limited():
    """A drag moves the view every frame; the line worth having is the last."""
    import run_foresight as rf

    poses = [(1.0, 2.0, 3.0, (0.0, 0.0, 0.0)),
             (9.0, 2.0, 3.0, (0.0, 0.0, 0.0))]
    pose_patch, log_patch = _watching(rf, poses)
    watch = rf.CameraWatch(min_interval=60.0)
    with pose_patch, log_patch:
        first = watch.poll()
        assert watch.poll() == first, "the limit should suppress the read"
        assert watch.poll(force=True) != first, "force must override it"


def test_camera_watch_survives_an_unreadable_camera():
    """Reporting the view is a convenience; it must never end a run."""
    from unittest import mock

    import run_foresight as rf

    with mock.patch.object(rf, "camera_pose",
                           side_effect=RuntimeError("no gui")):
        assert rf.CameraWatch().poll() is None


def test_fan_out_feeds_every_callback():
    """The frame hook holds one sink, but a GUI run has two things to do."""
    import run_foresight as rf

    seen = []
    assert rf._fan_out([]) is None
    single = (lambda: seen.append("a"))
    assert rf._fan_out([single]) is single
    rf._fan_out([lambda: seen.append("x"), lambda: seen.append("y")])()
    assert seen == ["x", "y"]


def test_a_crashed_episode_still_replays_what_it_captured():
    """The run you most want to look at is the one that broke.

    The replay used to sit after the try/finally, so an exception from the
    episode skipped it and closed the window on the traceback.
    """
    import inspect

    import run_foresight as rf

    source = inspect.getsource(rf.main)
    body = source[source.index("recorder = LazyRecorder"):]
    assert "except Exception" in body
    # The failure path replays before it re-raises, so the exit status still
    # reports the failure.
    replay_at = body.index("play_until_interrupted")
    raise_at = body.index("raise failure")
    assert replay_at < raise_at


def test_a_fresh_track_is_not_mistaken_for_absent():
    """ReplayTrack defines __len__, so an empty one is falsy.

    The frame sink used to be built with ``track.capture if track else None``,
    which therefore installed nothing: every track is empty when the sink is
    built, so no frame was ever captured and both the replay loop and the
    still renderer had nothing to work with.
    """
    import inspect

    import run_foresight as rf

    track = rf.ReplayTrack()
    assert len(track) == 0
    assert not track, "guarding the regression below, not the behaviour"

    source = inspect.getsource(rf.main)
    assert "track.capture if track is not None else None" in source, (
        "the frame sink must test the track against None, not its truthiness")
