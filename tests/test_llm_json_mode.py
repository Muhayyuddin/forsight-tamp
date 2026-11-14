"""OpenAI's json_object guard is a substring check, not a semantic one.

A prompt may specify its schema perfectly by example and still be refused with
a 400 before the model is consulted, because the literal word "json" never
appears. That is how the candidate-selection prompt -- the core of ForeSight's
LLM-guided selection -- silently failed every call on API models.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.llm_util import (  # noqa: E402
    JSON_MODE_REMINDER,
    _ensure_json_mentioned,
)

# Verbatim shape of the prompt that produced the 400s: schema by example only.
CANDIDATE_PROMPT = (
    "Select one hard-feasible candidate for the action template below.\n"
    "Action template: place(['disk_3_small'], {})\n"
    'Return exactly: {"Reasoning":"one short sentence","Candidate ID":"A02C001"}'
)


def test_the_candidate_prompt_really_lacks_the_keyword():
    """Guards the premise; if this ever fails the other tests prove nothing."""
    assert "json" not in CANDIDATE_PROMPT.lower()


def test_keyword_is_added_when_absent():
    patched = _ensure_json_mentioned(
        [{"role": "user", "content": CANDIDATE_PROMPT}])
    assert "json" in patched[0]["content"].lower()
    assert patched[0]["content"].startswith(CANDIDATE_PROMPT)


def test_prompt_already_naming_json_is_untouched():
    messages = [{"role": "user", "content": "Reply in JSON please."}]
    assert _ensure_json_mentioned(messages) is messages


def test_detection_ignores_case_and_looks_at_every_message():
    messages = [
        {"role": "system", "content": "Emit JSON only."},
        {"role": "user", "content": CANDIDATE_PROMPT},
    ]
    assert _ensure_json_mentioned(messages) is messages


def test_caller_messages_are_not_mutated():
    """The planner reuses its message list; editing in place would leak."""
    messages = [{"role": "user", "content": CANDIDATE_PROMPT}]
    _ensure_json_mentioned(messages)
    assert messages[0]["content"] == CANDIDATE_PROMPT


def test_reminder_lands_on_the_last_user_turn():
    messages = [
        {"role": "user", "content": CANDIDATE_PROMPT},
        {"role": "assistant", "content": "understood"},
        {"role": "user", "content": "now choose"},
    ]
    patched = _ensure_json_mentioned(messages)
    assert patched[2]["content"].endswith(JSON_MODE_REMINDER)
    assert patched[0]["content"] == CANDIDATE_PROMPT


def test_conversation_without_a_user_turn_still_gets_the_keyword():
    patched = _ensure_json_mentioned(
        [{"role": "system", "content": "be terse"}])
    assert "json" in patched[-1]["content"].lower()
