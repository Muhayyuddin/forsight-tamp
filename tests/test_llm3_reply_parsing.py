"""LLM3 must tolerate the reply formats its siblings already tolerate.

It parsed with strict json.loads while every other baseline used
safe_json_object. A model that wraps its JSON in a markdown fence therefore
failed for LLM3 alone: the same planner scored 70% on a model answering bare
and 0% on one that fences, for plans that were never inspected.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.common import safe_json_object  # noqa: E402

PLANNERS = [
    ROOT / "planners" / "llm_tamp_planner.py",
]
PLANNER = PLANNERS[0]
REPLY = '{"Reasoning": "r", "Full Plan": ["pick([\'a\'], {})"]}'


def test_no_live_planner_parses_a_reply_with_bare_json_loads():
    offending = []
    for path in PLANNERS:
        if not path.exists():
            continue
        text = path.read_text()
        # Catches both the one-line and wrapped-call spellings.
        flat = " ".join(text.split())
        if re.search(r"json\.loads\( self\.prompt_llm", flat):
            offending.append(str(path.relative_to(ROOT)))
    assert not offending, (
        "these planners parse a model reply with strict json.loads: "
        + ", ".join(offending))


def test_every_live_planner_uses_the_shared_tolerant_parser():
    missing = [
        str(p.relative_to(ROOT)) for p in PLANNERS
        if p.exists() and "safe_json_object" not in p.read_text()
    ]
    assert not missing, "not using safe_json_object: " + ", ".join(missing)


def test_a_fenced_reply_parses():
    assert safe_json_object(f"```json\n{REPLY}\n```")["Full Plan"]


def test_a_bare_reply_parses():
    assert safe_json_object(REPLY)["Full Plan"]


def test_a_reply_wrapped_in_prose_parses():
    assert safe_json_object(f"Here is the plan:\n{REPLY}\nDone.")["Full Plan"]


def test_a_reply_with_no_object_still_raises():
    """Tolerance must not extend to accepting a non-answer."""
    import pytest
    with pytest.raises(Exception):
        safe_json_object("I cannot help with that.")
