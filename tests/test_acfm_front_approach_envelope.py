"""A side-grasp candidate must keep the swept gripper volume clear.

ACFM's bounds, collision, reachability and support masks all test the object
footprint alone.  A horizontal approach sweeps the hand through the space
beside and in front of the target, so a candidate one object-width from an
already-placed neighbour satisfies every mask yet cannot be executed.  Those
cells score highest, because they look like compact placements, so a greedy
selection spent its entire rollout quota on them and the planning step
reported no feasible candidate at all.
"""
import numpy as np
import pytest

from acfm import ACFM


def _map_with_neighbour(neighbour_cx):
    acfm = ACFM.__new__(ACFM)
    acfm.target_dims = (0.055, 0.055, 0.07)
    acfm.target_aabb_z_offset = 0.0
    acfm.existing_footprints = [{
        "cx": neighbour_cx, "cy": 0.20,
        "hx": 0.0225, "hy": 0.0225,
        "z_min": 0.205, "z_max": 0.29,
    }]
    return acfm


def test_candidate_beside_a_neighbour_is_rejected():
    acfm = _map_with_neighbour(0.46)
    assert not acfm._front_approach_envelope_is_clear(0.52, 0.20, 0.2525), (
        "a candidate 0.06 m from the neighbour leaves no room for a hand that "
        "spans 0.108 m either side of the tool frame"
    )


def test_candidate_clear_of_the_hand_width_is_accepted():
    acfm = _map_with_neighbour(0.40)
    assert acfm._front_approach_envelope_is_clear(0.60, 0.20, 0.2525)


def test_a_neighbour_on_another_level_does_not_block():
    acfm = _map_with_neighbour(0.46)
    acfm.existing_footprints[0].update({"z_min": 0.045, "z_max": 0.15})
    assert acfm._front_approach_envelope_is_clear(0.52, 0.20, 0.2525), (
        "an object on a different shelf level shares no space with the hand"
    )


def test_the_approach_corridor_in_front_of_the_target_must_be_clear():
    """The hand reaches 0.133 m behind the tool frame, into the bay mouth."""
    acfm = _map_with_neighbour(0.52)
    acfm.existing_footprints[0].update({"cy": 0.09, "hy": 0.025})
    assert not acfm._front_approach_envelope_is_clear(0.52, 0.20, 0.2525)


@pytest.mark.parametrize("separation", [0.0, 0.05, 0.10])
def test_insufficient_separation_is_always_rejected(separation):
    acfm = _map_with_neighbour(0.50 - separation)
    assert not acfm._front_approach_envelope_is_clear(0.50, 0.20, 0.2525)


def test_the_three_dimensional_map_applies_the_same_envelope_check():
    """The shelf domain builds ACFM3D, which overrides candidate generation.

    The check originally lived only on the 2-D base class, so the shelf path
    that actually needs it never ran it.
    """
    import inspect

    from acfm import ACFM3D

    source = inspect.getsource(ACFM3D.get_parameter_candidates)
    assert "_front_approach_envelope_is_clear" in source, (
        "ACFM3D.get_parameter_candidates must apply the gripper envelope "
        "check; it overrides the base implementation entirely"
    )
    assert ACFM3D._front_approach_envelope_is_clear is (
        ACFM._front_approach_envelope_is_clear), (
        "both maps must share one definition of the hand envelope")
