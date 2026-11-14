"""Truncation must be counted, because downstream it is invisible.

When `max_tokens` stops generation, the reply arrives cut mid-structure. The
parser rejects it, the planner re-prompts, the budget drains, and the scenario
is recorded as a planning failure -- identical in every observable way to a
model that simply answered badly. `finish_reason` is the only thing that tells
the two apart, and nothing read it until this metric existed.

Measured twice before it was caught: Qwen3.6-35B-A3B returned EMPTY content at
a 1000-token cap (thinking consumed the allowance), and GLM-4.7-Flash returned
content cut mid-key at the same cap, scoring 4/9 on work it had not failed.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.metrics_util import trial_trace  # noqa: E402


def _response(finish_reason, content="ok", completion=3):
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    if finish_reason is not None:
        choice.finish_reason = finish_reason
    return SimpleNamespace(
        id="r1", model="m",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=completion,
                              total_tokens=5 + completion),
        choices=[choice],
    )


def _record(*responses):
    with trial_trace(max_llm_requests=None) as trace:
        for response in responses:
            index = trace.reserve_llm_request(
                messages=[{"role": "user", "content": "p"}],
                request_parameters={}, logical_call_id="c", retry_index=0)
            trace.record_llm_response(index, response)
        return trace, trace.summary()


def test_length_finish_reason_is_counted():
    _, summary = _record(_response("length"))
    assert summary["llm_truncated_responses"] == 1


def test_a_normal_stop_is_not_counted():
    _, summary = _record(_response("stop"))
    assert summary["llm_truncated_responses"] == 0


def test_the_counter_accumulates_across_calls():
    _, summary = _record(
        _response("length"), _response("stop"), _response("length"))
    assert summary["llm_truncated_responses"] == 2


def test_a_response_without_finish_reason_does_not_crash():
    """Local OpenAI-compatible servers do not all populate the field."""
    _, summary = _record(_response(None))
    assert summary["llm_truncated_responses"] == 0


def test_the_reason_is_kept_on_the_event_for_diagnosis():
    trace, _ = _record(_response("length"))
    assert trace.llm_events[0]["finish_reason"] == "length"


def test_truncation_is_reported_even_when_content_is_empty():
    """The Qwen signature: allowance spent on thinking, content comes back ''."""
    _, summary = _record(_response("length", content=""))
    assert summary["llm_truncated_responses"] == 1


def test_summary_always_exposes_the_key():
    """A missing key would read as zero truncations and hide the defect."""
    _, summary = _record()
    assert summary["llm_truncated_responses"] == 0
