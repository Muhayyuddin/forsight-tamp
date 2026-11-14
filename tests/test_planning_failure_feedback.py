"""A rejected skeleton must tell the next attempt why it was rejected.

Execution failures already reached the next prompt. Planning failures did not:
the loop cleared the feedback list, so every retry was issued against "No prior
failure feedback". The model had no reason to revise, returned a byte-identical
skeleton, and the cycle repeated until the call budget was exhausted.

Observed on TOH4_4DiskCylinder with gpt-4o-mini: four consecutive attempts, all
claiming no prior failure, all returning the same 8-action skeleton, all
rejected at `action 5 place(['disk_2_large'], {})`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_foresight import ForeSightTAMP  # noqa: E402


def _planner():
    """A bare instance: these helpers must not depend on a live simulator."""
    return ForeSightTAMP.__new__(ForeSightTAMP)


def test_no_failure_recorded_yields_no_feedback():
    assert _planner()._planning_failure_feedback() == []


def test_a_recorded_failure_is_rendered_for_the_prompt():
    planner = _planner()
    planner._record_planning_failure(
        "place(['disk_2_large'], {})",
        "action 5 was rejected during planning")
    feedback = planner._planning_failure_feedback()
    assert len(feedback) == 1
    action, entry = feedback[0]
    assert action == "place(['disk_2_large'], {})"
    assert "action 5 was rejected" in entry.motion_planner_feedback
    assert entry.action_success is False
    assert entry.goal_achieved is False


def test_the_reason_survives_the_prompt_formatting():
    """The prompt joins entries as "{action}: {motion_planner_feedback}"."""
    planner = _planner()
    planner._record_planning_failure("place(['x'], {})", "no candidate exists")
    rendered = "\n".join(
        f"{action}: {entry.motion_planner_feedback}"
        for action, entry in planner._planning_failure_feedback()
    ) or "No prior failure feedback."
    assert rendered != "No prior failure feedback."
    assert "no candidate exists" in rendered


def test_a_failure_without_a_template_is_still_legible():
    planner = _planner()
    planner._record_planning_failure(None, "could not construct a skeleton")
    action, entry = planner._planning_failure_feedback()[0]
    assert action == "skeleton"
    assert "None" not in f"{action}: {entry.motion_planner_feedback}"


def test_a_later_failure_replaces_an_earlier_one():
    """Only the current attempt's reason may be shown to the next attempt."""
    planner = _planner()
    planner._record_planning_failure("a", "first reason")
    planner._record_planning_failure("b", "second reason")
    action, entry = planner._planning_failure_feedback()[0]
    assert action == "b"
    assert "first reason" not in entry.motion_planner_feedback


def test_every_planning_failure_site_records_a_reason():
    """Guards against a new early return quietly reintroducing the bug."""
    import inspect
    source = inspect.getsource(ForeSightTAMP._plan_sequentially_with_acfm)
    lines = source.splitlines()
    returns = [i for i, line in enumerate(lines)
               if line.strip() == "return None"]
    assert returns, "expected early returns in the sequential planner"
    for index in returns:
        window = "\n".join(lines[max(0, index - 8):index])
        assert "_record_planning_failure" in window, (
            f"a 'return None' near line {index} of the sequential planner "
            "abandons the attempt without recording why")
