#!/usr/bin/env python3
"""
Run ForeSight-TAMP on a scenario with PyBullet GUI.

Usage:
    python3 run_foresight.py                           # default: A1_easy_few
    python3 run_foresight.py --scenario A2_medium_mixed
    python3 run_foresight.py --scenario A1_easy_few --no-gui
"""
import argparse
import json
import logging
import os
import select
import shutil
import sys
import tempfile
import time
from pathlib import Path

from main_foresight import ForeSightTAMP
from envs.pack_compact_env import PackCompactEnv
from envs.stacking_env import StackingEnv
from utils import replay_capture
from utils.llm_util import configure_llm_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_foresight")


def load_scenario(scenario_name: str) -> dict:
    """Load a scenario JSON and extract the task_config for env.reset()."""
    scenario_path = (
        Path(__file__).parent
        / "envs"
        / "task_instances"
        / "foresight"
        / f"{scenario_name}.json"
    )
    if not scenario_path.exists():
        logger.error(f"Scenario file not found: {scenario_path}")
        sys.exit(1)

    with open(scenario_path) as f:
        scenario = json.load(f)

    env_type = scenario.get("env_type", "packing")

    if env_type == "stacking":
        task_config = {
            "boxes": scenario["boxes"],
            "stacking_order": scenario["stacking_order"],
            "stacking_zone": scenario.get("stacking_zone"),
            "task": scenario.get("task"),
        }
    elif env_type == "shelf":
        task_config = {
            "shelf_config": scenario["shelf_config"],
            "boxes": scenario["boxes"],
            "task": scenario.get("task"),
        }
    elif env_type == "push_stacking":
        task_config = {
            "boxes": scenario["boxes"],
            "stacking_order": scenario["stacking_order"],
            "target_pose": scenario["target_pose"],
            "task": scenario.get("task"),
        }
    elif env_type == "hanoi":
        task_config = {
            "disks": scenario["disks"],
            "pegs": scenario["pegs"],
            "task": scenario.get("task"),
        }
    else:
        task_config = {
            "basket": scenario["basket"],
            "boxes": scenario["boxes"],
            "task_type": scenario.get("task_type", "packing"),
            "stacking_order": scenario.get("stacking_order"),
            "task": scenario.get("task"),
        }
    return task_config, scenario



def ensure_ffmpeg_on_path():
    """Make an ``ffmpeg`` binary discoverable, reporting whether one is.

    PyBullet's MP4 logger shells out to ``ffmpeg`` and silently writes nothing
    when it is missing. If PATH has none, the binary bundled with
    imageio-ffmpeg is used.
    """
    if shutil.which("ffmpeg"):
        return True
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        executable = Path(get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001 - absence is the expected case
        logger.debug("no bundled ffmpeg: %s", exc)
        return False
    if not executable.exists():
        return False
    shim = Path(tempfile.gettempdir()) / "foresight-ffmpeg"
    shim.mkdir(exist_ok=True)
    link = shim / "ffmpeg"
    if not link.exists():
        try:
            link.symlink_to(executable)
        except OSError as exc:  # noqa: BLE001
            logger.debug("could not link ffmpeg: %s", exc)
            return False
    os.environ["PATH"] = f"{shim}{os.pathsep}{os.environ.get('PATH', '')}"
    logger.info("using bundled ffmpeg at %s", executable)
    return True


def _start_recording(args, use_gui):
    """Begin an MP4 capture of the GUI, when one was asked for.

    PyBullet can only record what it is drawing, so this is a no-op (with a
    warning) without a GUI.
    """
    if not args.save_video:
        return None
    if not use_gui:
        logger.warning(
            "--save-video needs the GUI; nothing will be recorded headless")
        return None
    if not ensure_ffmpeg_on_path():
        logger.warning(
            "no ffmpeg on PATH and none bundled: PyBullet writes MP4 through "
            "ffmpeg and will silently record nothing. "
            "Install it with: pip install imageio-ffmpeg")
        return None
    import pybullet as p

    directory = Path(args.save_video)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.scenario}.mp4"
    try:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, str(path))
    except Exception as exc:  # noqa: BLE001 - recording is never fatal
        logger.warning("could not start recording: %s", exc)
        return None
    logger.info("recording -> %s", path)
    return {"log_id": log_id, "path": path}


class LazyRecorder:
    """Start the MP4 once the simulator is actually connected.

    The connection is made when the environment first resets, and logging
    cannot start before it, so the log is started from the first drawn frame.
    """

    def __init__(self, args, use_gui):
        self.args = args
        self.use_gui = use_gui
        self.handle = None
        self.restarts = 0

    def restart(self):
        """(Re)start the log, because the scene has just been rebuilt.

        resetSimulation silently terminates an active video log, and the
        replanning loop resets more than once per episode, so the log is
        restarted on each reset and the surviving file covers the last attempt.
        """
        self._stop_log()
        self.handle = _start_recording(self.args, self.use_gui)
        self.restarts += 1

    def _stop_log(self):
        """Close the current log without converting anything."""
        if not self.handle:
            return
        import pybullet as p

        try:
            p.stopStateLogging(self.handle["log_id"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not close the previous log: %s", exc)
        self.handle = None


class ReplayTrack:
    """Frames of the GUI animation, kept so the run can be replayed.

    A frame is the pose of every body plus every joint position, which is
    general enough to cover the arm, the objects it carries and anything a
    scenario adds without knowing what any of them are. Replay resets those
    states rather than re-simulating, so what the window shows afterwards is
    what it showed during the run, and no planning or physics is repeated.

    The MP4 and GIF are closed before the first replay starts, so they hold
    one pass however long the window keeps looping.
    """

    def __init__(self, stride=1):
        self.stride = max(1, int(stride))
        self.frames = []
        self._seen = 0
        # Frame indices at which the scene was rebuilt. Each rebuild starts a
        # new attempt, so the frames after the last one that has any are the
        # committed execution that ended the episode.
        self._boundaries = []

    def __len__(self):
        return len(self.frames)

    def new_segment(self):
        """The scene has just been rebuilt; later frames are a new attempt."""
        if len(self.frames) > (self._boundaries[-1] if self._boundaries else 0):
            self._boundaries.append(len(self.frames))

    def final_frames(self):
        """Frames of the last attempt that executed anything."""
        start = max([0] + [b for b in self._boundaries if b < len(self.frames)])
        return self.frames[start:]

    def capture(self):
        self._seen += 1
        if (self._seen - 1) % self.stride:
            return
        import pybullet as p

        frame = []
        for index in range(p.getNumBodies()):
            body = p.getBodyUniqueId(index)
            position, orientation = p.getBasePositionAndOrientation(body)
            joints = p.getNumJoints(body)
            values = ([state[0] for state in
                       p.getJointStates(body, list(range(joints)))]
                      if joints else [])
            frame.append((body, position, orientation, values))
        self.frames.append(frame)

    def _show(self, frame):
        import pybullet as p

        for body, position, orientation, values in frame:
            p.resetBasePositionAndOrientation(body, position, orientation)
            for joint, value in enumerate(values):
                p.resetJointState(body, joint, value)

    def write_stills(self, directory, stem, count, size, camera):
        """Render ``count`` frames of the captured run as high-resolution PNGs.

        Each frame is posed again from the stored body states and rendered
        with the given camera at the requested resolution, without
        re-simulating anything.

        Returns the paths written, in run order.
        """
        from PIL import Image

        if not self.frames:
            logger.warning("nothing was captured, so no stills were written")
            return []

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        width, height = size
        view, projection = self._camera(camera, width, height)

        last = len(self.frames) - 1
        # The last sample is the final frame, not near it: the point of the
        # closing tile is the state the episode actually finished in.
        positions = [min(last, round(last * index / max(1, count - 1)))
                     for index in range(count)]

        written = []
        for index, position in enumerate(positions, start=1):
            self._show(self.frames[position])
            pixels = self._render(view, projection, width, height)
            path = directory / f"{stem}_{index}.png"
            Image.fromarray(pixels).save(path, optimize=True)
            written.append(path)
        logger.info("wrote %d stills at %dx%d to %s",
                    len(written), width, height, directory)
        return written

    @staticmethod
    def _camera(camera, width, height):
        import pybullet as p

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(camera["target"]),
            distance=camera["distance"], yaw=camera["yaw"],
            pitch=camera["pitch"], roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(
            fov=camera["fov"], aspect=width / height, nearVal=0.05,
            farVal=12.0)
        return view, projection

    @staticmethod
    def _render(view, projection, width, height):
        """The posed scene as an RGB array, from the given camera."""
        import numpy as np
        import pybullet as p

        try:
            shot = p.getCameraImage(
                width, height, viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        except Exception as exc:  # noqa: BLE001 - fall back, do not fail
            logger.debug("hardware render failed (%s); using the "
                         "software renderer", exc)
            shot = p.getCameraImage(
                width, height, viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER)
        rendered_w, rendered_h = shot[0], shot[1]
        return np.reshape(np.asarray(shot[2], dtype=np.uint8),
                          (rendered_h, rendered_w, 4))[:, :, :3]

    def write_video(self, path, size, camera, fps=60, supersample=2,
                    hold=(1.0, 2.0), crf=16, gif_path=None, gif_loop=True):
        """Render the committed execution as a high-resolution MP4.

        Unlike the window recording, which holds the whole planning process,
        this renders only the frames of the final attempt, each posed again
        and rendered offscreen. Frames are rendered at ``supersample`` times
        the size and downscaled, which smooths the edges the renderer leaves
        aliased. ``hold`` keeps the first and last frame on screen for that
        many seconds, so the start and the goal state can be read.
        """
        import subprocess

        import numpy as np
        from PIL import Image

        frames = self.final_frames()
        if not frames:
            logger.warning("nothing was captured, so no video was written")
            return None
        if not ensure_ffmpeg_on_path():
            logger.warning("no ffmpeg available; no video was written")
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        factor = max(1, int(supersample))
        view, projection = self._camera(camera, width * factor,
                                        height * factor)
        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        gif = []
        written = 0
        last = len(frames) - 1
        for index, frame in enumerate(frames):
            self._show(frame)
            image = Image.fromarray(self._render(
                view, projection, width * factor, height * factor))
            if factor > 1:
                # A box average of each factor-by-factor block: the cheap,
                # exact downscale for an integer supersample.
                image = image.reduce(factor)
            data = np.asarray(image).tobytes()
            repeats = 1
            if index == 0:
                repeats += round(hold[0] * fps)
            if index == last:
                repeats += round(hold[1] * fps)
            for _ in range(repeats):
                encoder.stdin.write(data)
                if gif_path is not None and written % GIF_FRAME_STRIDE == 0:
                    gif.extend(_gif_frames([np.asarray(image)], stride=1))
                written += 1
        encoder.stdin.close()
        if encoder.wait() != 0:
            logger.warning("ffmpeg failed; %s may be incomplete", path)
            return None
        logger.info("rendered %d execution frames (%.1f s) at %dx%d -> %s",
                    len(frames), written / fps, width, height, path)

        if gif_path is not None and gif:
            import imageio.v3 as iio

            iio.imwrite(gif_path, gif, **_gif_write_kwargs(gif_loop))
            logger.info("saved gif -> %s", gif_path)
        return path

    def play_until_interrupted(self, watch=None, period=1.0 / 60.0, pause=0.6):
        """Loop the captured run in the open window until Ctrl+C."""
        if not self.frames:
            logger.warning(
                "nothing was captured to replay; the window will just stay open")
            _wait_for_enter(watch)
            return
        logger.info("replaying %d frames on a loop \u2014 press Ctrl+C to stop",
                    len(self.frames))
        try:
            while True:
                for frame in self.frames:
                    self._show(frame)
                    # Every frame, so a view framed mid-pass is reported while
                    # the mouse is still on it rather than a pass later.
                    if watch is not None:
                        watch.poll()
                    time.sleep(period)
                time.sleep(pause)
        except KeyboardInterrupt:
            logger.info("replay stopped")
            if watch is not None:
                watch.poll(force=True)


def camera_pose():
    """The viewer's current camera, as (yaw, pitch, distance, target).

    getDebugVisualizerCamera returns twelve fields; the four that place the
    camera sit at the end, behind the view and projection matrices.
    """
    import pybullet as p

    info = p.getDebugVisualizerCamera()
    return (float(info[8]), float(info[9]), float(info[10]),
            tuple(float(value) for value in info[11]))


def format_camera_pose(pose):
    """The pose as the call that reproduces it, so it can be pasted back."""
    yaw, pitch, distance, target = pose
    return ("p.resetDebugVisualizerCamera(cameraDistance=%.2f, "
            "cameraYaw=%.1f, cameraPitch=%.1f, "
            "cameraTargetPosition=[%.3f, %.3f, %.3f])"
            % (distance, yaw, pitch, target[0], target[1], target[2]))


class CameraWatch:
    """Report the GUI camera whenever the viewer drags it somewhere new.

    Polled, not subscribed: PyBullet raises no camera-changed event, so the
    pose is read on frames the window is drawing anyway and whenever the run
    is otherwise idle.

    Reports are rate-limited, and poses are compared as formatted strings so
    a change below the printed precision is not reported.
    """

    def __init__(self, min_interval=0.3):
        self.min_interval = min_interval
        self.last_line = None
        self.last_poll = 0.0

    def poll(self, force=False):
        """Log the pose if it has changed since the last line. Never raises."""
        now = time.monotonic()
        if not force and now - self.last_poll < self.min_interval:
            return self.last_line
        self.last_poll = now
        try:
            line = format_camera_pose(camera_pose())
        except Exception as exc:  # noqa: BLE001 - never worth failing a run over
            logger.debug("could not read the camera pose: %s", exc)
            return self.last_line
        if line != self.last_line:
            self.last_line = line
            logger.info("camera -> %s", line)
        return line


def _fan_out(callbacks):
    """One sink that feeds several, since only one can be installed."""
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    def sink():
        for callback in callbacks:
            callback()
    return sink


def _wait_for_enter(watch=None):
    """Hold the window open, reporting camera moves while we wait.

    A plain input() would block camera polling, so stdin is waited on with a
    timeout and the camera is polled between checks.
    """
    prompt = "Press Enter to close the GUI...\n"
    if watch is None:
        try:
            input(prompt.strip())
        except EOFError:
            pass
        return
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        while True:
            watch.poll()
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if not sys.stdin.readline():
                    return  # stdin closed
                return
    except (EOFError, KeyboardInterrupt):
        pass


def _read_video(iio, path):
    """Read an MP4 back with whichever imageio backend is installed.

    Tries ``pyav`` and then ``FFMPEG`` (the backend imageio-ffmpeg provides).
    """
    errors = []
    for plugin in ("pyav", "FFMPEG"):
        try:
            return iio.imread(path, plugin=plugin)
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{plugin}: {exc}")
    raise RuntimeError("; ".join(errors))


#: Every eighth frame at 560 px keeps a GIF small while the motion stays
#: readable; the MP4 beside it stays full quality.
GIF_MAX_WIDTH = 560
GIF_FRAME_STRIDE = 8


def _gif_frames(frames, max_width=GIF_MAX_WIDTH, stride=GIF_FRAME_STRIDE):
    """Thin and shrink a clip into something worth sending."""
    import numpy as np
    from PIL import Image

    thinned = []
    for frame in frames[::stride]:
        image = Image.fromarray(np.asarray(frame))
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.LANCZOS)
        thinned.append(np.asarray(image))
    return thinned


def _gif_write_kwargs(loop, fps=10):
    """Frame timing and repeat settings for the written GIF.

    ``loop=0`` is the GIF Netscape block's "repeat forever". Playing once is
    the absence of that block, not ``loop=1`` -- most viewers read 1 as one
    *additional* repeat and play the clip twice -- so the key is omitted
    rather than set.

    Timing is given as ``duration`` in milliseconds per frame, since current
    imageio ignores ``fps``.
    """
    kwargs = {"duration": 1000.0 / fps}
    if loop:
        kwargs["loop"] = 0
    return kwargs


def _stop_recording(recording, args):
    if not recording:
        return
    import pybullet as p

    try:
        p.stopStateLogging(recording["log_id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not stop recording cleanly: %s", exc)
        return
    path = recording["path"]
    logger.info("saved recording -> %s", path)
    if not args.save_gif:
        return
    try:
        import imageio.v3 as iio

        frames = _read_video(iio, path)
        gif_path = path.with_suffix(".gif")
        looping = getattr(args, "gif_loop", True)
        iio.imwrite(gif_path, _gif_frames(frames),
                    **_gif_write_kwargs(looping))
        logger.info("saved gif -> %s (%.1f MB, %s)", gif_path,
                    gif_path.stat().st_size / 1e6,
                    "repeats until stopped" if looping else "plays once")
    except Exception as exc:  # noqa: BLE001 - the MP4 is already safe on disk
        logger.warning(
            "kept the MP4 but could not build a GIF (%s); convert it with "
            "ffmpeg if you need one", exc)


def build_parser():
    parser = argparse.ArgumentParser(description="Run ForeSight-TAMP")
    parser.add_argument(
        "--scenario",
        default="A1_easy_few",
        help="Scenario name (without .json), e.g. A1_easy_few",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable PyBullet GUI (headless mode)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=20,
        help="Maximum LLM planning iterations",
    )
    parser.add_argument(
        "--model",
        "--llm-model",
        dest="llm_model",
        default=None,
        help="LLM model name. Examples: gpt-4o, qwen3:8b",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible API base URL. Example: http://localhost:11434/v1 for Ollama",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the selected endpoint. Use 'ollama' for local Ollama.",
    )
    parser.add_argument(
        "--llm-response-format",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to send OpenAI response_format. Auto disables it for local Ollama endpoints.",
    )
    parser.add_argument(
        "--save-video",
        metavar="DIR",
        default=None,
        help="Record the GUI session to DIR as MP4 (requires GUI).",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Also convert the recording to an animated GIF.",
    )
    parser.add_argument(
        "--no-gif-loop",
        dest="gif_loop",
        action="store_false",
        help="Write the GIF to play once. The default repeats it until the "
             "viewer stops it.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit non-zero when the goal is not reached. Lets a caller "
             "recording demonstrations tell a solved run from a recording of "
             "a failure, which looks identical on disk.",
    )
    parser.add_argument(
        "--close-when-done",
        action="store_true",
        help="Close the window as soon as the run ends: no replay loop, no "
             "waiting for Enter. Needed to record several scenarios in a row, "
             "which would otherwise stop at the first one.",
    )
    parser.add_argument(
        "--replay-loop",
        action="store_true",
        help="After the run, replay it in the GUI on a loop until Ctrl+C. "
             "The recording is closed first, so the MP4 and GIF still hold a "
             "single pass.",
    )
    parser.add_argument(
        "--play-traj",
        action="store_true",
        help="Animate robot trajectories in GUI",
    )
    parser.add_argument(
        "--render-video",
        metavar="DIR",
        help=("Render the committed execution of the final attempt as a "
              "high-resolution MP4 in DIR, without the planning process. "
              "Uses the still camera. Written only when the goal is reached."),
    )
    parser.add_argument(
        "--video-size", default="1920x1080", metavar="WxH",
        help="Pixel size of the rendered video (default: 1920x1080).",
    )
    parser.add_argument(
        "--stills",
        metavar="DIR",
        help=("Render high-resolution PNGs of the finished run into DIR. "
              "Independent of --save-video: these are rendered from the "
              "captured states, not decoded from the recording."),
    )
    parser.add_argument(
        "--still-count", type=int, default=3,
        help="How many stills to spread over the run (default: 3).",
    )
    parser.add_argument(
        "--still-size", default="1600x1200", metavar="WxH",
        help="Pixel size of each still (default: 1600x1200).",
    )
    parser.add_argument(
        "--still-distance", type=float, default=1.90,
        help="Still camera distance in metres (default: 1.90). With a 45 "
             "degree field of view this holds the table, the arm and the "
             "tallest scene, the shelf, with a little margin.",
    )
    parser.add_argument(
        "--still-yaw", type=float, default=40.0,
        help="Still camera yaw in degrees (default: 40, as in the window).",
    )
    parser.add_argument(
        "--still-pitch", type=float, default=-45.0,
        help="Still camera pitch in degrees (default: -45).",
    )
    parser.add_argument(
        "--still-target-z", type=float, default=0.12,
        help="Height of the still camera's look-at point above the table.",
    )
    parser.add_argument(
        "--still-fov", type=float, default=45.0,
        help="Still camera vertical field of view in degrees (default: 45).",
    )
    parser.add_argument(
        "--disable",
        nargs="*",
        default=[],
        choices=["acfm", "rehearsal", "cascade"],
        help="Disable specific modules for ablation",
    )
    return parser


def _still_camera(args):
    """Camera for the rendered stills, framed on the table."""
    from envs.pb_env import PybulletEnv

    return {
        "distance": args.still_distance,
        "yaw": args.still_yaw,
        "pitch": args.still_pitch,
        "fov": args.still_fov,
        "target": (PybulletEnv.TABLE_CENTER_X, PybulletEnv.TABLE_CENTER_Y,
                   args.still_target_z),
    }


class _EglRenderer:
    """Load PyBullet's EGL renderer once the headless server exists.

    The connection is made inside the first environment reset, so this rides
    the scene sink; the plugin then serves getCameraImage on the GPU.
    """

    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.loaded = True
        try:
            import pkgutil

            import pybullet as p

            egl = pkgutil.get_loader("eglRenderer")
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            logger.info("rendering offscreen with the EGL plugin")
        except Exception as exc:  # noqa: BLE001 - falls back to TinyRenderer
            logger.warning("no EGL renderer (%s); rendering will be slow", exc)


def _write_stills(track, args):
    try:
        width, height = (int(part) for part in args.still_size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--still-size wants WxH, got {args.still_size!r}")
    track.write_stills(args.stills, args.scenario, args.still_count,
                       (width, height), _still_camera(args))


def main():
    args = build_parser().parse_args()

    response_format = None
    if args.llm_response_format == "on":
        response_format = "1"
    elif args.llm_response_format == "off":
        response_format = "0"
    configure_llm_runtime(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        response_format=response_format,
    )

    use_gui = not args.no_gui

    task_config, scenario = load_scenario(args.scenario)
    env_type = scenario.get("env_type", "packing")
    logger.info(
        f"Running scenario: {scenario.get('scenario_id', args.scenario)} "
        f"- {scenario.get('scenario_name', '')} [{env_type}]"
    )
    logger.info(f"GUI: {use_gui}, Modules disabled: {args.disable or 'none'}")
    if args.llm_model or args.llm_base_url:
        logger.info(
            "LLM override: model=%s, base_url=%s",
            args.llm_model or "<default>",
            args.llm_base_url or "<OpenAI default>",
        )

    # Select appropriate environment
    if env_type == "stacking":
        env = StackingEnv()
    elif env_type == "shelf":
        from envs.shelf_env import ShelfEnv
        env = ShelfEnv()
    elif env_type == "push_stacking":
        from envs.push_stacking_env import PushStackingEnv
        env = PushStackingEnv()
    elif env_type == "hanoi":
        from envs.hanoi_env import TowerOfHanoiEnv
        env = TowerOfHanoiEnv()
    else:
        env = PackCompactEnv()

    foresight = ForeSightTAMP(
        env=env,
        max_llm_calls=args.max_llm_calls,
        disable_modules=args.disable,
        use_gui=use_gui,
        save_dir=f"results/{args.scenario}",
    )

    # Always animate in GUI mode unless user explicitly passes --no-gui
    play_traj = args.play_traj or use_gui
    track = None
    watch = CameraWatch() if use_gui else None
    if args.close_when_done:
        args.replay_loop = False
    # Stills and rendered video use the same captured states as the replay.
    if (args.replay_loop or args.stills or args.render_video) and use_gui:
        track = ReplayTrack()
    elif args.render_video:
        # Headless rendering uses EGL on the GPU, far faster than the GUI's
        # software GL.
        track = ReplayTrack()
    elif args.replay_loop:
        logger.warning("--replay-loop needs the GUI; there is nothing to replay")
    elif args.stills:
        logger.warning("--stills needs the GUI; no states were captured")
    # Recording, capture and camera polling ride the scene and frame hooks.
    recorder = LazyRecorder(args, use_gui)
    replay_capture.set_scene_sink(_fan_out(
        [recorder.restart]
        + ([track.new_segment] if track is not None else [])
        + ([_EglRenderer().load] if args.render_video and not use_gui
           else [])))
    # `is not None`, not truthiness: an empty ReplayTrack is falsy.
    replay_capture.set_sink(_fan_out(
        [callback for callback in
         (track.capture if track is not None else None,
          watch.poll if watch is not None else None)
         if callback]))

    failure = None
    try:
        result = foresight.run_episode(task_config, play_traj=play_traj)
    except Exception as exc:  # noqa: BLE001 - re-raised once the window closes
        # Frames captured up to the failure are still replayed; the exception
        # is re-raised afterwards so the exit status reports the failure.
        failure = exc
        result = None
        logger.exception("the episode failed; replaying what was captured")
    finally:
        replay_capture.set_sink(None)
        replay_capture.set_scene_sink(None)
        _stop_recording(recorder.handle, args)

    if failure is not None:
        if use_gui and not args.close_when_done:
            watch.poll(force=True)
            if track is not None:
                track.play_until_interrupted(watch)
            else:
                _wait_for_enter(watch)
        env.destroy()
        raise failure

    logger.info("=" * 50)
    logger.info(f"Goal achieved: {result['goal_achieved']}")
    logger.info(f"LLM calls: {result['num_llm_calls']}")
    logger.info(f"MP calls: {result['num_mp_calls']}")
    logger.info(f"Wasted real actions: {result['wasted_real_actions']}")
    logger.info(f"Rehearsal saves: {result['rehearsal_saves']}")
    logger.info(f"Cascade depth: {result['cascade_depth']}")
    logger.info(f"ACFM accuracy: {result['acfm_accuracy']}%")
    logger.info("=" * 50)

    # Stills and video are written only for runs that reached the goal.
    if args.stills and track is not None and result["goal_achieved"]:
        _write_stills(track, args)
    elif args.stills and track is not None:
        logger.warning("goal not reached, so no stills were written")
    if args.render_video and track is not None and result["goal_achieved"]:
        try:
            width, height = (int(part) for part in
                             args.video_size.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--video-size wants WxH, got {args.video_size!r}")
        directory = Path(args.render_video)
        track.write_video(
            directory / f"{args.scenario}.mp4", (width, height),
            _still_camera(args),
            gif_path=(directory / f"{args.scenario}.gif"
                      if args.save_gif else None),
            gif_loop=getattr(args, "gif_loop", True))
    elif args.render_video and track is not None:
        logger.warning("goal not reached, so no video was rendered")

    if use_gui and not args.close_when_done:
        watch.poll(force=True)
        if track is not None:
            track.play_until_interrupted(watch)
        else:
            _wait_for_enter(watch)
    elif use_gui:
        watch.poll(force=True)

    env.destroy()
    if args.require_success and not result["goal_achieved"]:
        logger.warning("goal not reached; reporting failure to the caller")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
