"""A step that admits nothing may be infeasible, or merely over-constrained.

Both report "no feasible candidate", and from outside the planner they are
indistinguishable -- which is how a map that rejected every workable
placement went unnoticed for a whole study. The distinction is available at
the moment of failure: if any candidate the map rejected was then executed by
the simulator, the step is not infeasible and the masks are too tight. These
tests pin that the planner records it.
"""
import logging

import pytest

from main_foresight import ForeSightTAMP


def _outcome(cid, map_ok, sim_ok):
    return {"candidate_id": cid,
            "endpoint_map_feasible": map_ok,
            "ground_truth_feasible": sim_ok}


def _witnesses(outcomes):
    """The predicate the planner applies, isolated for testing."""
    return [c for c in outcomes
            if c.get("ground_truth_feasible")
            and not c.get("endpoint_map_feasible")]


def test_a_rejected_but_executable_candidate_is_a_witness():
    assert _witnesses([_outcome("A02C001", False, True)])


def test_a_step_whose_candidates_all_failed_the_simulator_is_not_flagged():
    """Nothing worked, so the step may genuinely be infeasible."""
    assert not _witnesses([_outcome("A02C001", False, False),
                           _outcome("A02C002", True, False)])


def test_an_admitted_candidate_that_failed_is_not_a_witness():
    """The map admitting something the rollout rejects is the safe direction."""
    assert not _witnesses([_outcome("A02C001", True, False)])


def test_every_witness_is_counted_not_just_the_first():
    outcomes = [_outcome("A02C001", False, True),
                _outcome("A02C002", False, True),
                _outcome("A02C003", True, False)]
    assert len(_witnesses(outcomes)) == 2


def test_the_warning_names_a_witness_so_it_can_be_rechecked():
    """A count alone is unactionable; the message must identify a candidate."""
    import inspect

    source = inspect.getsource(ForeSightTAMP)
    branch = source[source.index("if not feasible_candidates:"):]
    branch = branch[:branch.index("No accepted transition candidate")]
    assert "Relaxation counterexample" in branch
    assert 'witnesses[0].get("candidate_id")' in branch
    assert "too tight" in branch


def test_the_monitor_is_wired_into_the_exhaustion_branch():
    """The counter must be incremented where the planner gives up on a step."""
    import inspect

    source = inspect.getsource(ForeSightTAMP)
    branch = source[source.index("if not feasible_candidates:"):]
    branch = branch[:branch.index("No accepted transition candidate")]
    assert "relaxation_attributable_exhaustions" in branch
    assert "ground_truth_feasible" in branch
    assert "endpoint_map_feasible" in branch
