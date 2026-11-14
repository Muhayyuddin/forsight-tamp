"""Helpers for keeping computational PyBullet state changes out of the GUI."""

from functools import wraps

from pybullet_planning import LockRenderer


def suppress_gui_rendering(function):
    """Run a state-mutating computation without displaying intermediate states."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with LockRenderer():
            return function(*args, **kwargs)

    return wrapped
