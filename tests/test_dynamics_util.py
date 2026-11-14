import pytest

from utils.dynamics_util import apply_configured_dynamics


class FakePyBullet:
    def __init__(self):
        self.calls = []

    def changeDynamics(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_applies_only_explicit_scenario_dynamics():
    fake = FakePyBullet()
    applied = apply_configured_dynamics(
        7,
        {
            "mass": 0.4,
            "lateral_friction": 0.8,
            "linear_damping": 0.2,
            "unrelated": 123,
        },
        pybullet_module=fake,
    )
    assert applied == {
        "mass": 0.4,
        "lateralFriction": 0.8,
        "linearDamping": 0.2,
    }
    assert fake.calls == [((7, -1), applied)]


def test_rejects_negative_physical_values():
    fake = FakePyBullet()
    with pytest.raises(ValueError, match="mass"):
        apply_configured_dynamics(
            1, {"mass": -0.1}, pybullet_module=fake
        )
    with pytest.raises(ValueError, match="lateral_friction"):
        apply_configured_dynamics(
            1, {"lateral_friction": -0.1}, pybullet_module=fake
        )
    assert fake.calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass", float("nan"), "finite"),
        ("rolling_friction", float("inf"), "finite"),
        ("linear_damping", -0.1, "nonnegative"),
        ("angular_damping", -0.1, "nonnegative"),
        ("restitution", 1.1, "between 0 and 1"),
        ("mass", True, "finite"),
    ],
)
def test_rejects_invalid_dynamics_values(field, value, message):
    fake = FakePyBullet()
    with pytest.raises(ValueError, match=message):
        apply_configured_dynamics(
            1, {field: value}, pybullet_module=fake
        )
    assert fake.calls == []
