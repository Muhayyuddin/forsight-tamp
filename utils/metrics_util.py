"""Trial-scoped benchmark instrumentation.

This module provides one small, dependency-free trace shared by the LLM,
simulator, rehearsal, and evaluation layers.  It counts HTTP retries and keeps
committed execution separate from shadow rehearsal.

The trace is stored in :mod:`contextvars`, so instrumentation remains local to
the current trial if evaluation is later parallelised with threads or async
tasks.  It intentionally does not contain credentials.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator


class LLMRequestBudgetExceeded(RuntimeError):
    """Raised before an API request would exceed the per-trial hard budget."""


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class BenchmarkTrace:
    """Mutable trace for one method--scenario trial."""

    max_llm_requests: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    capture_llm_payloads: bool = True
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    durations: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_events: list[dict[str, Any]] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)

    def increment(self, name: str, amount: float = 1, phase: str | None = None) -> None:
        key = f"{phase}.{name}" if phase else name
        self.counters[key] += amount

    def add_duration(self, name: str, seconds: float, phase: str | None = None) -> None:
        key = f"{phase}.{name}" if phase else name
        self.durations[key] += max(0.0, float(seconds))

    def record_event(self, kind: str, **details: Any) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 6),
                "kind": kind,
                **_jsonable(details),
            }
        )

    def reserve_llm_request(
        self,
        *,
        messages: Any,
        request_parameters: dict[str, Any],
        logical_call_id: str,
        retry_index: int,
    ) -> int:
        """Reserve and record one actual call to ``chat.completions.create``.

        The budget check happens before the SDK is invoked.  The OpenAI client
        is configured with SDK retries disabled, so this count is the actual
        number of HTTP requests initiated by this process.
        """
        # The budget bounds how many times the planner may consult the model,
        # so it counts logical calls: a transport retry is not charged, so the
        # result does not depend on server stability.  Every attempt is still
        # counted in llm_api_requests for cost accounting.
        used = int(self.counters.get("llm_api_requests", 0))
        logical = int(self.counters.get("llm_logical_calls", 0))
        if (not retry_index and self.max_llm_requests is not None
                and logical >= self.max_llm_requests):
            self.increment("llm_budget_rejections")
            raise LLMRequestBudgetExceeded(
                f"Hard LLM request budget exhausted: "
                f"{logical}/{self.max_llm_requests}"
            )

        self.increment("llm_api_requests")
        if retry_index:
            self.increment("llm_transport_retries")
        else:
            self.increment("llm_logical_calls")
        event = {
            "request_index": used,
            "logical_call_id": logical_call_id,
            "retry_index": retry_index,
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 6),
            "messages_sha256": _sha256_json(messages),
            "request_parameters": _jsonable(request_parameters),
            "status": "started",
        }
        if self.capture_llm_payloads:
            event["messages"] = _jsonable(messages)
        self.llm_events.append(event)
        return len(self.llm_events) - 1

    def record_llm_response(self, event_index: int, response: Any) -> None:
        event = self.llm_events[event_index]
        event["status"] = "succeeded"
        event["completed_elapsed_seconds"] = round(
            time.monotonic() - self.started_monotonic, 6
        )
        event["response_id"] = getattr(response, "id", None)
        model = getattr(response, "model", None)
        event["resolved_model"] = model
        if model:
            models = self.counters.setdefault("_resolved_models", [])
            if isinstance(models, list) and model not in models:
                models.append(model)

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        event["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.increment("llm_prompt_tokens", prompt_tokens)
        self.increment("llm_completion_tokens", completion_tokens)
        self.increment("llm_total_tokens", total_tokens)

        choices = getattr(response, "choices", None) or []
        content = None
        finish_reason = None
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)
            finish_reason = getattr(choices[0], "finish_reason", None)
        event["finish_reason"] = finish_reason
        # "length" means the cap stopped generation, not the model. Downstream
        # a cut reply is simply unparseable, which looks identical to a model
        # that answered badly -- so it has to be counted here or not at all.
        if finish_reason == "length":
            self.increment("llm_truncated_responses")
        event["response_sha256"] = _sha256_json(content)
        if self.capture_llm_payloads:
            event["response_content"] = content

    def record_llm_error(self, event_index: int, exc: BaseException) -> None:
        event = self.llm_events[event_index]
        event["status"] = "failed"
        event["completed_elapsed_seconds"] = round(
            time.monotonic() - self.started_monotonic, 6
        )
        event["error_type"] = type(exc).__name__
        event["error"] = str(exc)

    def summary(self) -> dict[str, Any]:
        counters = dict(self.counters)
        resolved_models = counters.pop("_resolved_models", [])
        prompt_tokens = int(counters.get("llm_prompt_tokens", 0))
        completion_tokens = int(counters.get("llm_completion_tokens", 0))
        cost = None
        if (
            self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        ):
            cost = (
                prompt_tokens * self.input_cost_per_million
                + completion_tokens * self.output_cost_per_million
            ) / 1_000_000.0

        component_counts = {
            key: int(value) if float(value).is_integer() else value
            for key, value in counters.items()
            if not key.startswith("llm_")
        }

        def across_phases(metric: str) -> int | float:
            total = sum(
                value
                for key, value in component_counts.items()
                if key == metric or key.endswith(f".{metric}")
            )
            return int(total) if float(total).is_integer() else total

        def phase_value(phase: str, metric: str) -> int | float:
            value = component_counts.get(f"{phase}.{metric}", 0)
            return int(value) if float(value).is_integer() else value

        return {
            "metrics_schema": "foresight-benchmark-trace-v1",
            "llm_api_requests": int(counters.get("llm_api_requests", 0)),
            "llm_transport_retries": int(counters.get("llm_transport_retries", 0)),
            "llm_logical_calls": int(counters.get("llm_logical_calls", 0)),
            "llm_truncated_responses": int(
                counters.get("llm_truncated_responses", 0)),
            "llm_budget_rejections": int(counters.get("llm_budget_rejections", 0)),
            "llm_prompt_tokens": prompt_tokens,
            "llm_completion_tokens": completion_tokens,
            "llm_total_tokens": int(counters.get("llm_total_tokens", 0)),
            "llm_estimated_cost_usd": None if cost is None else round(cost, 8),
            "resolved_models": list(resolved_models),
            "ik_calls": across_phases("ik_calls"),
            "collision_checks": across_phases("collision_checks"),
            "birrt_calls": across_phases("birrt_calls"),
            "motion_planner_calls": across_phases("motion_planner_calls"),
            "candidate_rollout_attempts": across_phases(
                "candidate_rollout_attempts"),
            "candidate_rollout_passes": across_phases(
                "candidate_rollout_passes"),
            "matched_rollout_domain_candidates": phase_value(
                "rollout_control", "domain_candidates"),
            "matched_rollout_validation_quota": phase_value(
                "rollout_control", "validation_quota"),
            "matched_rollout_validation_attempted": phase_value(
                "rollout_control", "validation_attempted"),
            "matched_rollout_validation_passed": phase_value(
                "rollout_control", "validation_passed"),
            "simulator_steps": across_phases("simulator_steps"),
            "simulator_primitive_attempts": across_phases(
                "simulator_primitive_attempts"
            ),
            "metric_counts": component_counts,
            "metric_durations_seconds": {
                key: round(value, 6) for key, value in self.durations.items()
            },
            "trace_elapsed_seconds": round(
                time.monotonic() - self.started_monotonic, 6
            ),
        }

    def export(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "events": list(self.events),
            "llm_events": list(self.llm_events),
        }


_CURRENT_TRACE: contextvars.ContextVar[BenchmarkTrace | None] = contextvars.ContextVar(
    "foresight_benchmark_trace", default=None
)
_CURRENT_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "foresight_benchmark_phase", default=None
)


def current_trace() -> BenchmarkTrace | None:
    return _CURRENT_TRACE.get()


def current_phase() -> str | None:
    return _CURRENT_PHASE.get()


@contextlib.contextmanager
def trial_trace(
    *,
    max_llm_requests: int | None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    capture_llm_payloads: bool = True,
) -> Iterator[BenchmarkTrace]:
    trace = BenchmarkTrace(
        max_llm_requests=max_llm_requests,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        capture_llm_payloads=capture_llm_payloads,
    )
    token = _CURRENT_TRACE.set(trace)
    phase_token = _CURRENT_PHASE.set("execution")
    try:
        yield trace
    finally:
        _CURRENT_PHASE.reset(phase_token)
        _CURRENT_TRACE.reset(token)


@contextlib.contextmanager
def metric_phase(name: str) -> Iterator[None]:
    token = _CURRENT_PHASE.set(name)
    try:
        yield
    finally:
        _CURRENT_PHASE.reset(token)


@contextlib.contextmanager
def metric_timer(name: str, phase: str | None = None) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        trace = current_trace()
        if trace is not None:
            trace.add_duration(name, time.monotonic() - started, phase or current_phase())


def increment_metric(name: str, amount: float = 1, phase: str | None = None) -> None:
    trace = current_trace()
    if trace is not None:
        trace.increment(name, amount, phase or current_phase())


def record_metric_event(kind: str, **details: Any) -> None:
    trace = current_trace()
    if trace is not None:
        trace.record_event(kind, phase=current_phase(), **details)
