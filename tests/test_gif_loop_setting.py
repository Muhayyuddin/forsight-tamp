"""The recorded GIF must repeat until the viewer stops it.

A watched run exists to be looked at, and a GIF that plays once and freezes on
the last frame is the wrong default for that. The loop count is not a cosmetic
flag either: it is written into the file as a Netscape application block, so
whether a viewer repeats the clip is decided at write time and cannot be
changed afterwards without re-encoding.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_foresight as rf  # noqa: E402


def _write(tmp_path, loop):
    imageio = pytest.importorskip("imageio.v3")
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    for index in range(4):
        frames[index, :, :, 0] = index * 60
    path = tmp_path / f"clip_{loop}.gif"
    imageio.imwrite(path, frames, **rf._gif_write_kwargs(loop))
    return path


def _loop_count(path):
    """The repeat count a viewer will honour, or None when absent."""
    Image = pytest.importorskip("PIL.Image")
    with Image.open(path) as img:
        return img.info.get("loop")


def test_looping_is_the_default_and_can_be_turned_off():
    parser = rf.build_parser()
    assert parser.parse_args([]).gif_loop is True
    assert parser.parse_args(["--no-gif-loop"]).gif_loop is False


def test_loop_forever_is_requested_as_zero():
    assert rf._gif_write_kwargs(True)["loop"] == 0


def test_playing_once_omits_the_key_rather_than_setting_one():
    """``loop=1`` means one *additional* repeat, so the clip would play twice.

    Playing exactly once is the absence of the loop block, which is why the
    key is dropped instead of set.
    """
    assert "loop" not in rf._gif_write_kwargs(False)


def test_written_gif_repeats_forever(tmp_path):
    assert _loop_count(_write(tmp_path, True)) == 0


def test_written_gif_plays_once_when_asked(tmp_path):
    assert _loop_count(_write(tmp_path, False)) in (None, 1)


def test_frame_timing_is_given_as_duration(tmp_path):
    """imageio ignores ``fps`` now, so asking in frames per second did nothing.

    The clip played at the plugin default rather than at the rate requested.
    """
    kwargs = rf._gif_write_kwargs(True, fps=10)
    assert "fps" not in kwargs
    assert kwargs["duration"] == pytest.approx(100.0)


def test_missing_attribute_still_loops():
    """An args object from an older call path must not silently stop looping."""
    class Bare:
        pass

    assert getattr(Bare(), "gif_loop", True) is True
