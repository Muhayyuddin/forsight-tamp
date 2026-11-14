"""Recording a run must fail loudly, not leave an empty directory.

Both of the bugs this file pins down were silent. PyBullet writes MP4 by
shelling out to ffmpeg and, with no binary on PATH, returns a normal log id
and writes nothing. And the log was started before run_episode, which is where
the simulator actually connects, so it refused with "Not connected to physics
server" and the run carried on recording nothing at all.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_foresight as rf  # noqa: E402


class _Args:
    def __init__(self, **kwargs):
        self.scenario = "A1_easy_few"
        self.save_video = None
        self.save_gif = False
        self.gif_loop = True
        self.__dict__.update(kwargs)


def test_recording_restarts_every_time_the_scene_is_rebuilt(tmp_path):
    """resetSimulation terminates a running log, and replanning resets again.

    Starting once and hoping leaves a 1.8 kB header: the log is killed by the
    next reset and nothing says so. Restarting on each reset means the file
    that survives covers the attempt that actually ran.
    """
    recorder = rf.LazyRecorder(_Args(save_video=str(tmp_path)), use_gui=True)
    with mock.patch.object(rf, "_start_recording",
                           side_effect=[{"log_id": 1}, {"log_id": 2}]) as start:
        with mock.patch.object(recorder, "_stop_log") as stop:
            recorder.restart()
            recorder.restart()
    assert start.call_count == 2
    assert stop.call_count == 2, "the dead log must be closed before reopening"
    assert recorder.handle == {"log_id": 2}
    assert recorder.restarts == 2


def test_a_failed_start_leaves_a_handle_that_stops_cleanly(tmp_path):
    recorder = rf.LazyRecorder(_Args(save_video=str(tmp_path)), use_gui=True)
    with mock.patch.object(rf, "_start_recording", return_value=None):
        recorder.restart()
    assert recorder.handle is None
    rf._stop_recording(recorder.handle, _Args())  # must not raise


def test_no_video_is_requested_when_none_was_asked_for():
    recorder = rf.LazyRecorder(_Args(save_video=None), use_gui=True)
    recorder.restart()
    assert recorder.handle is None


def test_the_scene_hook_is_inert_until_someone_listens():
    """The benchmark must not gain a recorder by loading this module."""
    from utils import replay_capture

    replay_capture.set_scene_sink(None)
    replay_capture.scene_ready()  # must not raise
    seen = []
    replay_capture.set_scene_sink(lambda: seen.append(1))
    replay_capture.scene_ready()
    replay_capture.set_scene_sink(None)
    replay_capture.scene_ready()
    assert seen == [1]


def test_resetting_the_scene_announces_itself():
    """Without this call the recorder never learns the log was killed."""
    source = (ROOT / "envs" / "pb_env.py").read_text()
    assert "replay_capture.scene_ready()" in source


def test_recording_refuses_headless_rather_than_writing_nothing(tmp_path):
    assert rf._start_recording(_Args(save_video=str(tmp_path)),
                               use_gui=False) is None


def test_missing_ffmpeg_blocks_the_recording_instead_of_faking_it(tmp_path):
    """Without the guard PyBullet accepts the request and writes no file."""
    with mock.patch.object(rf, "ensure_ffmpeg_on_path", return_value=False):
        assert rf._start_recording(_Args(save_video=str(tmp_path)),
                                   use_gui=True) is None


def test_a_system_ffmpeg_is_used_as_is():
    with mock.patch.object(rf.shutil, "which", return_value="/usr/bin/ffmpeg"):
        assert rf.ensure_ffmpeg_on_path() is True


def test_ffmpeg_is_found_one_way_or_another():
    """imageio-ffmpeg ships a binary, so this should hold on a dev machine."""
    assert rf.ensure_ffmpeg_on_path() is True


def test_video_is_read_with_whichever_backend_exists():
    """The code asked for pyav by name, which imageio does not install."""
    calls = []

    class FakeIio:
        def imread(self, path, plugin=None):
            calls.append(plugin)
            if plugin == "pyav":
                raise ImportError("the pyav plugin is not installed")
            return "frames"

    assert rf._read_video(FakeIio(), "clip.mp4") == "frames"
    assert calls == ["pyav", "FFMPEG"]


def test_an_unreadable_video_reports_every_backend_it_tried():
    class FakeIio:
        def imread(self, path, plugin=None):
            raise OSError(f"{plugin} said no")

    with pytest.raises(RuntimeError) as exc:
        rf._read_video(FakeIio(), "clip.mp4")
    assert "pyav" in str(exc.value) and "FFMPEG" in str(exc.value)


def test_close_when_done_turns_off_the_replay_loop():
    """A batch of recordings would otherwise stop at the first scenario."""
    args = rf.build_parser().parse_args(["--close-when-done", "--replay-loop"])
    assert args.close_when_done is True
    assert args.replay_loop is True  # main() is what reconciles them
    source = __import__("inspect").getsource(rf.main)
    assert "if args.close_when_done:" in source
    assert "args.replay_loop = False" in source


def test_an_unsolved_goal_is_reported_to_the_caller():
    """A recording of a failure is indistinguishable on disk from a success.

    Without this the batch would keep a clip of the arm not solving the task
    and call the scenario done.
    """
    import inspect

    source = inspect.getsource(rf.main)
    assert "args.require_success and not result[\"goal_achieved\"]" in source
    assert "return 1" in source
    parser = rf.build_parser()
    assert parser.parse_args([]).require_success is False
    assert parser.parse_args(["--require-success"]).require_success is True


def test_the_batch_asks_for_success_and_retries():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_recorder", ROOT / "scripts" / "record_paper_scenarios.py")
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    class Args:
        model = "gpt-4o-mini"
        max_llm_calls = 20
        llm_base_url = None
        llm_api_key = "ollama"
        video_size = "1920x1080"
        with_stills = False
        stills_dir = Path("/tmp/stills")
        still_count = 3
        still_size = "1600x1200"

    built = recorder.command("A1_easy_few", Args(), Path("/tmp/out"))
    assert "--require-success" in built, "a failed run must not count as done"
    assert "--close-when-done" in built, "a batch cannot wait on Ctrl+C"
    assert "--render-video" in built and "--save-gif" in built
    assert built[built.index("--render-video") + 1] == "/tmp/out"
    assert "--stills" not in built, "stills are opt-in"

    Args.with_stills = True
    built = recorder.command("A1_easy_few", Args(), Path("/tmp/out"))
    assert built[built.index("--stills") + 1] == "/tmp/stills"
    assert "--no-gui" not in built, "stills need the GUI"
    assert built[built.index("--still-size") + 1] == "1600x1200"
