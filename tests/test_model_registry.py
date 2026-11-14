"""The model roster must stay internally consistent.

A five-model study is only comparable if every run records which model
answered and under what limits.  These tests pin the invariants that make the
registry safe to loop over.
"""
import pytest

from configs.model_registry import (
    MODEL_REGISTRY,
    RECOMMENDED_ORDER,
    get_model,
    model_keys,
    provenance,
)


def test_every_entry_validates():
    for key in MODEL_REGISTRY:
        get_model(key)


def test_unknown_key_names_the_alternatives():
    with pytest.raises(KeyError, match="Unknown model key"):
        get_model("no-such-model")


def test_recommended_order_covers_every_study_model():
    """Smoke-test entries prove the plumbing and must stay out of the study."""
    study = {
        key for key, entry in MODEL_REGISTRY.items()
        if not entry.get("smoke_test_only")
    }
    assert set(RECOMMENDED_ORDER) == study
    assert model_keys()[0] == "gpt-4o-mini", (
        "the cheapest strong API model should be the pilot")
    assert model_keys()[-1] == "llama33-70b", (
        "the tightest VRAM fit should run last so it can be descoped")


def test_smoke_models_are_excluded_from_the_default_loop():
    smoke = [
        key for key, entry in MODEL_REGISTRY.items()
        if entry.get("smoke_test_only")
    ]
    assert smoke, "expected at least one smoke-test entry"
    for key in smoke:
        assert key not in model_keys(), (
            f"{key} would contaminate a study loop")
        assert key in model_keys(include_smoke=True)


@pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
def test_local_models_declare_an_endpoint_and_free_pricing(key):
    entry = get_model(key)
    if entry["kind"] != "local":
        return
    assert entry["base_url"], "a local model needs a reachable endpoint"
    assert entry["input_cost_per_million"] == 0.0
    assert entry["output_cost_per_million"] == 0.0
    assert entry["response_format"] == "off", (
        "many OpenAI-compatible local servers reject response_format")


@pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
def test_local_timeouts_exceed_the_hosted_default(key):
    """A locally served model is far slower than a hosted one."""
    entry = get_model(key)
    if entry["kind"] == "local":
        assert entry["timeout_seconds"] >= 600.0


def test_llama_context_stays_small_enough_for_its_quantisation():
    """42.5 GB of weights on a 48 GB card leaves little room for KV cache.

    Measured by loading, not by arithmetic: at 16384 the server reports
    3%/97% CPU/GPU -- it spills, and partial CPU inference is far slower than
    the smaller window costs. At 12288 it stays fully resident at 46 GB with
    2.3 GB spare, which still holds a 1.3k prompt beside an 8k completion.
    """
    entry = get_model("llama33-70b")
    assert entry["context_tokens"] <= 12288, (
        "measured: 16384 spills to CPU on this card")


def test_prompt_fits_every_context_with_room_to_grow():
    """A measured ForeSight prompt is ~2,080 tokens before feedback accrues."""
    measured_prompt_tokens = 2080
    for key in MODEL_REGISTRY:
        entry = get_model(key)
        assert entry["context_tokens"] > measured_prompt_tokens, (
            f"{key} cannot even hold one prompt")


def test_provenance_records_the_right_identity_per_kind():
    api = provenance("gpt-4o-mini")
    assert "expected_resolved_model" in api, (
        "a hosted model is pinned by its dated snapshot")
    assert "gguf_file" not in api

    local = provenance("llama33-70b")
    assert local["gguf_file"], (
        "a local model has no dated snapshot, so the weights file is its "
        "identity")
    assert "quantisation" in local


def test_reasoning_models_get_a_larger_completion_allowance():
    """A reasoning model spends the allowance on thinking before answering.

    Measured on Qwen3.6-35B-A3B: at 1000 tokens a planning prompt stopped for
    length with an EMPTY content field, so every plan failed to parse and the
    model looked incapable of following instructions. At 4000 it returned valid
    JSON. The allowance covers reasoning tokens, so it is not a free parameter.
    """
    for key, entry in MODEL_REGISTRY.items():
        if entry.get("reasoning_model"):
            assert entry["max_tokens"] >= 4000, (
                f"{key} thinks before answering and needs headroom for both")


def test_every_model_declares_whether_it_reasons():
    for key, entry in MODEL_REGISTRY.items():
        assert "reasoning_model" in entry, (
            f"{key} must state this: it determines the completion allowance")
        assert isinstance(entry["reasoning_model"], bool)


def test_no_model_asks_for_more_completion_than_its_api_allows():
    """A cap above the model's own ceiling is rejected, not clamped.

    gpt-3.5-turbo refuses max_tokens above 4096 with invalid_value, so a
    blanket cap silently failed every one of its calls -- 200 rejections and
    an arm that scored zero before the cause was visible.
    """
    api_ceilings = {"gpt-3.5-turbo": 4096}
    for key, ceiling in api_ceilings.items():
        assert MODEL_REGISTRY[key]["max_tokens"] <= ceiling, (
            f"{key} allows at most {ceiling} completion tokens")


def test_every_completion_cap_clears_the_wordiest_baseline():
    """Measured: ViLaIn-TAMP averages ~860 tokens, GLM's baselines ~3900."""
    for key, entry in MODEL_REGISTRY.items():
        if entry.get("smoke_test_only"):
            continue
        assert entry["max_tokens"] >= 4000, (
            f"{key} would truncate the verbose baselines")
