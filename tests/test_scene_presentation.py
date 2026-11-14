"""The studio scenes have to look right, because they are what gets recorded.

Each of these pins a defect that was visible in a rendered frame and invisible
everywhere else: a shelf standing half off the table, PyBullet's checkered
plane showing through the studio floor, and a default light that left every
scene flat and grey.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.pb_env import PybulletEnv  # noqa: E402

SCENARIOS = ROOT / "envs" / "task_instances" / "foresight"


def _table_bounds():
    x_half = PybulletEnv.TABLE_WIDTH_X / 2.0
    y_half = PybulletEnv.TABLE_LENGTH_Y / 2.0
    return ((PybulletEnv.TABLE_CENTER_X - x_half,
             PybulletEnv.TABLE_CENTER_X + x_half),
            (PybulletEnv.TABLE_CENTER_Y - y_half,
             PybulletEnv.TABLE_CENTER_Y + y_half))


def _shelf_scenarios():
    for path in sorted(SCENARIOS.glob("*.json")):
        config = json.loads(path.read_text()).get("shelf_config")
        if config:
            yield path.stem, config


def test_there_are_shelf_scenarios_to_check():
    assert list(_shelf_scenarios()), "the glob found nothing to test"


def test_every_shelf_stands_entirely_on_the_table():
    """A shelf overhanging the edge is floating in mid-air in the picture."""
    (x_min, x_max), (y_min, y_max) = _table_bounds()
    for name, config in _shelf_scenarios():
        half_w = config["width"] / 2.0
        half_d = config["depth"] / 2.0
        wall = config.get("wall_thickness", 0.01)
        assert config["x"] - half_w >= x_min, f"{name} hangs off the back"
        assert config["x"] + half_w <= x_max, f"{name} hangs off the front"
        assert config["y"] - half_d >= y_min, f"{name} hangs off the right"
        # The back wall sits a wall thickness beyond the boards.
        assert config["y"] + half_d + wall <= y_max, f"{name} hangs off the left"


def test_no_shelf_sits_on_top_of_the_robot_base():
    """Fitting a shelf by shifting it can push it into the arm's own column."""
    for name, config in _shelf_scenarios():
        near_x = config["x"] - config["width"] / 2.0
        near_y = abs(config["y"]) - config["depth"] / 2.0
        distance = (near_x ** 2 + max(near_y, 0.0) ** 2) ** 0.5
        assert distance > 0.15, f"{name} crowds the robot base ({distance:.3f} m)"


def test_the_studio_lays_its_own_floor():
    """changeVisualShape does not reliably clear the imported plane's texture.

    The GUI keeps drawing it, tinted by whatever colour was requested, so the
    checker grid shows through. The fix has to be a different surface, not a
    different colour on the same one.
    """
    source = (ROOT / "envs" / "pb_env.py").read_text()
    assert "FLOOR_SIZE" in source
    assert "studio_floor = create_box" in source
    assert "rgbaColor=(0, 0, 0, 0)" in source, "the imported plane must be hidden"


def test_the_floor_slab_covers_more_than_the_camera_can_see():
    assert PybulletEnv.FLOOR_SIZE >= 6.0


def test_the_scene_is_lit_rather_than_left_on_the_default():
    assert len(PybulletEnv.LIGHT_POSITION) == 3
    assert PybulletEnv.LIGHT_POSITION[2] > 1.0, "the light must be above the table"
    source = (ROOT / "envs" / "pb_env.py").read_text()
    assert "configureDebugVisualizer(lightPosition=" in source
