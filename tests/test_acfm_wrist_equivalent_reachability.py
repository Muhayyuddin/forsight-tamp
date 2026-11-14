"""The map must test every wrist the controller is willing to use.

A parallel jaw closes on the same two faces at theta and at theta+pi, so the
controller retries a failed top placement with the flipped wrist. The map
tested only the first representation, which made it disagree with the very
executor it is supposed to predict: for an object grasped with a rotated
wrist, every cell inside the container came back unreachable while the
simulator placed it successfully. Endpoint and rollout feasibility are
combined with AND, so the map's verdict vetoed placements that physically
work and the planner reported no feasible candidate for the step.
"""
import numpy as np

from acfm import ACFM
from envs.pb_env import PyBulletRobot


def test_top_mode_offers_the_controllers_wrist_equivalents():
    _, orientations = ACFM._contact_ee_poses(0.5, 0.0, 0.1, "top")
    expected = [
        tuple(float(c) for c in q)
        for q in PyBulletRobot._wrist_equivalent_orientations(0.0)
    ]
    assert len(orientations) == 1 + len(expected), (
        "the map should try the nominal wrist and every equivalent the "
        "controller would retry")
    assert [tuple(float(c) for c in q) for q in orientations[1:]] == expected


def test_nominal_orientation_is_tried_first():
    """Order matters: the controller stops at the first wrist that plans."""
    _, orientations = ACFM._contact_ee_poses(0.5, 0.0, 0.1, "top")
    nominal = orientations[0]
    assert np.allclose(nominal, ACFM._contact_ee_pose(0.5, 0.0, 0.1, "top")[1])


def test_front_mode_keeps_its_single_orientation():
    """The controller's flip retry is specific to the top approach."""
    _, orientations = ACFM._contact_ee_poses(0.5, 0.2, 0.25, "front")
    assert len(orientations) == 1


def test_positions_are_unchanged_by_the_extra_orientations():
    for mode in ("top", "front"):
        position, _ = ACFM._contact_ee_poses(0.5, 0.1, 0.2, mode)
        assert position == ACFM._contact_ee_pose(0.5, 0.1, 0.2, mode)[0]


class _Probe:
    """Stands in for the IK pair, recording what was asked and in what order."""

    def __init__(self, accept):
        self.accept = accept
        self.asked = []

    def _calculate_ik_with_limits(self, robot_id, ee_link, pos, orn, **kwargs):
        self.asked.append(tuple(orn))
        return [0.0]

    def _ik_solution_is_valid(self, robot_id, ee_link, pos, orn, ik):
        return tuple(orn) == self.accept


def test_returns_the_first_orientation_that_verifies():
    probe = _Probe(accept=(1.0, 0.0, 0.0, 0.0))
    found = ACFM._first_reachable_orientation(
        probe, 0, 0, [0.0, 0.0, 0.1],
        [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
    assert found == (1.0, 0.0, 0.0, 0.0)
    assert len(probe.asked) == 1, (
        "a cell the controller reaches on the first wrist must not pay for "
        "the alternatives")


def test_falls_through_to_a_later_orientation():
    probe = _Probe(accept=(0.0, 1.0, 0.0, 0.0))
    found = ACFM._first_reachable_orientation(
        probe, 0, 0, [0.0, 0.0, 0.1],
        [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
    assert found == (0.0, 1.0, 0.0, 0.0)
    assert len(probe.asked) == 2


def test_reports_unreachable_when_no_orientation_verifies():
    probe = _Probe(accept=(9.0, 9.0, 9.0, 9.0))
    assert ACFM._first_reachable_orientation(
        probe, 0, 0, [0.0, 0.0, 0.1],
        [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]) is None


def test_an_ik_exception_does_not_abort_the_remaining_orientations():
    class _Raises(_Probe):
        def _calculate_ik_with_limits(self, robot_id, ee_link, pos, orn, **kw):
            self.asked.append(tuple(orn))
            if len(self.asked) == 1:
                raise RuntimeError("solver blew up")
            return [0.0]

    probe = _Raises(accept=(0.0, 1.0, 0.0, 0.0))
    found = ACFM._first_reachable_orientation(
        probe, 0, 0, [0.0, 0.0, 0.1],
        [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
    assert found == (0.0, 1.0, 0.0, 0.0)


class _Footprint:
    """A map whose only job is to answer the oriented-bounds question."""

    def __init__(self, dims, reference_yaw):
        self.target_dims = dims
        self.target_aabb_offset = [0.0, 0.0]
        self.target_reference_yaw = reference_yaw
        self.target_aabb_z_offset = 0.0
        self.existing_footprints = []
        self.bounds = {"x_min": 0.375, "x_max": 0.705,
                       "y_min": -0.135, "y_max": 0.135}


# The A2 grocery box, measured: 0.0558 along x and 0.1153 along y as it lies,
# and grasped with the wrist turned a quarter turn.
A2_BOX = (0.0558, 0.1153, 0.1476)
QUARTER = np.pi / 2.0


def test_a_rotated_grasp_lands_the_object_turned_by_that_grasp():
    """Commanding theta=0 on a quarter-turn grasp lands it across, not along.

    Measured in the simulator: picking the box and placing it at theta=0
    produces an AABB of 0.1156 x 0.0564, the transpose of how it started.
    A y-centre of -0.092 is therefore comfortably inside the container, and
    the map used to reject it because it tested the untransposed footprint.
    """
    turned = _Footprint(A2_BOX, reference_yaw=QUARTER)
    assert ACFM._oriented_candidate_is_endpoint_feasible(
        turned, 0.468, -0.092, 0.0), (
            "at theta=0 the 0.0558 m side runs along y, so a centre 0.043 m "
            "from the wall fits")


def test_the_same_pose_is_rejected_when_the_object_lies_along_y():
    """The identical placement genuinely does not fit the other way round."""
    square = _Footprint(A2_BOX, reference_yaw=0.0)
    assert not ACFM._oriented_candidate_is_endpoint_feasible(
        square, 0.468, -0.092, 0.0), (
            "with the 0.1153 m side along y the footprint reaches -0.1497, "
            "past the -0.135 wall")


def test_a_square_grasp_is_unaffected_by_the_change():
    """Every object in the other scenarios is grasped square."""
    square = _Footprint(A2_BOX, reference_yaw=0.0)
    for x, y, theta in ((0.54, 0.0, 0.0), (0.54, 0.0, QUARTER),
                        (0.40, 0.12, 0.0), (0.69, -0.12, QUARTER)):
        turned_by_zero = _Footprint(A2_BOX, reference_yaw=0.0)
        assert (ACFM._oriented_candidate_is_endpoint_feasible(
                    square, x, y, theta)
                is ACFM._oriented_candidate_is_endpoint_feasible(
                    turned_by_zero, x, y, theta))


def test_quarter_turn_grasp_swaps_which_thetas_fit():
    """The two grasps accept opposite orientations at the same point."""
    turned = _Footprint(A2_BOX, reference_yaw=QUARTER)
    square = _Footprint(A2_BOX, reference_yaw=0.0)
    point = (0.468, -0.092)
    assert ACFM._oriented_candidate_is_endpoint_feasible(turned, *point, 0.0)
    assert not ACFM._oriented_candidate_is_endpoint_feasible(
        turned, *point, QUARTER)
    assert not ACFM._oriented_candidate_is_endpoint_feasible(square, *point, 0.0)
    assert ACFM._oriented_candidate_is_endpoint_feasible(
        square, *point, QUARTER)
