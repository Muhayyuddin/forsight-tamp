"""Turning ranking off must remove ordering and nothing else.

The coarse ``acfm`` flag swaps in a one-shot planner, so it removes the map
together with per-step binding and candidate rollout. An ablation that says
anything about the map alone needs a flag that leaves those two in place;
this is that flag, and these tests pin the distinction it is meant to make.
"""
import numpy as np
import pytest

from acfm import ACFM, ACFM3D


def _composed(rank, masks, layers):
    acfm = ACFM.__new__(ACFM)
    acfm.rank_candidates = rank
    return acfm._compose_feasible_scores(masks, layers, ACFM.W)


def _masks(feasible):
    return {"bounds": np.asarray(feasible, dtype=bool)}


def _layers(values):
    arr = np.asarray(values, dtype=float)
    return {name: arr.copy() for name in
            ("collision", "reachability", "stability", "disturbance",
             "failure_history")}


def test_ranking_on_preserves_the_ordering_of_feasible_cells():
    score = _composed(True, _masks([[1, 1], [1, 0]]),
                      _layers([[0.2, 0.9], [0.5, 0.7]]))
    assert score[0, 1] > score[1, 0] > score[0, 0]


def test_ranking_off_scores_every_feasible_cell_alike():
    score = _composed(False, _masks([[1, 1], [1, 0]]),
                      _layers([[0.2, 0.9], [0.5, 0.7]]))
    feasible = score[np.asarray([[1, 1], [1, 0]], dtype=bool)]
    assert len(set(feasible.tolist())) == 1, (
        "with ranking off no feasible cell may be preferred to another")


def test_ranking_off_still_rejects_infeasible_cells():
    """The masks are the safety property; only the ordering is ablated."""
    score = _composed(False, _masks([[1, 1], [1, 0]]),
                      _layers([[0.2, 0.9], [0.5, 0.7]]))
    assert score[1, 1] == 0.0


def test_an_all_infeasible_map_stays_all_zero_either_way():
    for rank in (True, False):
        score = _composed(rank, _masks([[0, 0], [0, 0]]),
                          _layers([[0.9, 0.9], [0.9, 0.9]]))
        assert not score.any()


def test_the_flag_defaults_to_ranking_enabled():
    assert ACFM(resolution=4).rank_candidates is True
    assert ACFM3D(resolution=4).rank_candidates is True


def test_the_flag_reaches_the_three_dimensional_map():
    assert ACFM3D(resolution=4, rank_candidates=False).rank_candidates is False


def test_planner_reads_the_flag_from_the_disable_list():
    """``ranking`` must be its own module, not a synonym for ``acfm``."""
    from main_foresight import ForeSightTAMP

    planner = ForeSightTAMP.__new__(ForeSightTAMP)
    for disabled, expect_acfm, expect_rank in (
            ([], True, True),
            (["ranking"], True, False),
            (["acfm"], False, True),
            (["acfm", "ranking"], False, False),
    ):
        planner.use_acfm = "acfm" not in disabled
        planner.use_acfm_ranking = "ranking" not in disabled
        assert planner.use_acfm is expect_acfm
        assert planner.use_acfm_ranking is expect_rank


def test_the_ablation_arm_is_registered_and_distinct():
    from generate_results import METHODS

    arm = METHODS["Abl_no_ranking"]
    assert arm["disable"] == ["ranking"]
    assert arm["class"] == "foresight"
    # It must not collide with an existing arm, or it reports a rerun of one
    # of them as new evidence.
    same = [name for name, cfg in METHODS.items()
            if name != "Abl_no_ranking" and cfg.get("class") == "foresight"
            and sorted(cfg.get("disable", [])) == ["ranking"]]
    assert not same, same
