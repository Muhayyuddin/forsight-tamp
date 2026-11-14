"""A frame hook for replaying a finished run in the GUI.

The simulator animates a trajectory by setting joint positions and sleeping;
nothing in that loop keeps a history, so once a run ends the window holds only
its final state. This module lets a caller install a sink that is invoked once
per rendered frame, which is enough to replay the run afterwards.

Nothing installs a sink by default. When none is installed ``frame()`` is a
single global comparison and returns, so the executed path is identical with
and without this hook. The benchmark never installs one.
"""

_SINK = None


def set_sink(sink):
    """Install a per-frame callable, or clear it with ``None``.

    Returns the previous sink so a caller can restore it.
    """
    global _SINK
    previous = _SINK
    _SINK = sink
    return previous


def active():
    return _SINK is not None


def frame():
    """Offer the current simulator state to the installed sink, if any."""
    if _SINK is not None:
        _SINK()


_SCENE_SINK = None


def set_scene_sink(sink):
    """Install a callable invoked each time the scene finishes resetting.

    Separate from the frame sink because it answers a different question:
    not "what does the window show now" but "the world has just been rebuilt".
    A recorder needs that, since resetSimulation silently terminates an
    active video log and the replanning loop resets more than once.
    """
    global _SCENE_SINK
    previous = _SCENE_SINK
    _SCENE_SINK = sink
    return previous


def scene_ready():
    """Announce that the scene has been rebuilt, if anyone is listening."""
    if _SCENE_SINK is not None:
        _SCENE_SINK()
