"""Declarative roster of planner language models for the benchmark.

A run is pinned to one entry here.  Keeping endpoint, identifier, context
length, timeout and pricing in one table means the five-model study is a loop
over ``MODEL_REGISTRY`` rather than five divergent command lines, and it gives
the run manifest a single place to record which model actually answered.

Two kinds of entry:

``api``
    A hosted endpoint.  Reproducibility is pinned by ``expected_resolved_model``
    -- the dated snapshot the provider reports back.

``local``
    An OpenAI-compatible server on this machine, normally Ollama at
    ``http://localhost:11434/v1``.  There is no dated snapshot to pin, so
    provenance is the GGUF file name, its SHA-256 and the quantisation.  Fill
    ``gguf_sha256`` in once the weights are on disk; the validation step
    records it into the manifest.

Costs are US dollars per million tokens, used only to report what a run spent.
Local models are free but still report real token counts, so cross-model
tables stay uniform.
"""

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ``max_tokens`` covers reasoning tokens as well as the answer; a reasoning
# model that exhausts it while still thinking returns an EMPTY answer. The caps
# are set well above what the wordiest baseline emits, so truncation does not
# penalise the comparison methods more than ForeSight. Prompts peak near 1.5k
# tokens against a 16k context on the smallest model.
STANDARD_MAX_TOKENS = 8000
# gpt-3.5-turbo rejects any request above this with an invalid_value
# error; the limit belongs to the model, not to the protocol.
GPT35_MAX_COMPLETION_TOKENS = 4096
REASONING_MAX_TOKENS = 8000
# Non-reasoning models with long replies, such as GLM-4.7-Flash.
VERBOSE_MAX_TOKENS = 8000

MODEL_REGISTRY = {
    # ---------------------------------------------------------------- API ---
    "gpt-4o-mini": {
        "kind": "api",
        "model": "gpt-4o-mini",
        "label": "GPT-4o-mini",
        "base_url": None,
        "api_key": None,          # resolved from key file or environment
        "response_format": "auto",
        "context_tokens": 128_000,
        "timeout_seconds": 60.0,
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "expected_resolved_model": None,   # set to the dated snapshot when known
        "max_tokens": STANDARD_MAX_TOKENS,
        "reasoning_model": False,
    },
    "gpt-3.5-turbo": {
        "kind": "api",
        "model": "gpt-3.5-turbo",
        "label": "GPT-3.5-turbo",
        "base_url": None,
        "api_key": None,
        "response_format": "auto",
        "context_tokens": 16_385,
        "timeout_seconds": 60.0,
        "input_cost_per_million": 0.50,
        "output_cost_per_million": 1.50,
        "expected_resolved_model": None,
        "max_tokens": GPT35_MAX_COMPLETION_TOKENS,
        "reasoning_model": False,
    },

    # -------------------------------------------------------------- LOCAL ---
    # The Llama entry keeps a small context: at Q4_K_M the weights alone are
    # 42.5 GB of the 48 GB card, so the KV cache has to stay small.
    "qwen36-35b-a3b": {
        "kind": "local",
        "model": "qwen36-35b-a3b-q5km",
        "label": "Qwen3.6-35B-A3B",
        "base_url": OLLAMA_BASE_URL,
        "api_key": "ollama",
        "response_format": "off",
        "context_tokens": 65_536,
        "timeout_seconds": 600.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "gguf_file": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
        "gguf_sha256": "c13ce26253ea334df472bd8fbd2d6da66d8a41195c17f6fcbf44c4d20ece0932",
        "quantisation": "UD-Q5_K_M",
        "disk_gb": 26.5,
        "max_tokens": REASONING_MAX_TOKENS,
        "reasoning_model": True,
    },
    "glm47-flash": {
        "kind": "local",
        "model": "glm47-flash-q5km",
        "label": "GLM-4.7-Flash",
        "base_url": OLLAMA_BASE_URL,
        "api_key": "ollama",
        "response_format": "off",
        "context_tokens": 65_536,
        "timeout_seconds": 600.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "gguf_file": "GLM-4.7-Flash-Q5_K_M.gguf",
        "gguf_sha256": "167848c4b460887d7a52c44c378d3996a0d38b3302ac2045494dc8b50869a893",
        "quantisation": "Q5_K_M",
        "disk_gb": 21.4,
        "max_tokens": VERBOSE_MAX_TOKENS,
        "reasoning_model": False,
    },
    # Validation-only entry: a small model that is already on disk, used to
    # prove the local path end to end (server, endpoint, prompt assembly, plan
    # parsing, metric recording) without waiting on a 90 GB download.  It is
    # deliberately excluded from RECOMMENDED_ORDER and is not part of the study.
    "qwen3-8b-smoke": {
        "kind": "local",
        "model": "qwen3:8b",
        "label": "Qwen3 8B (smoke test only)",
        "base_url": OLLAMA_BASE_URL,
        "api_key": "ollama",
        "response_format": "off",
        "context_tokens": 32_768,
        "timeout_seconds": 600.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "smoke_test_only": True,
        "max_tokens": REASONING_MAX_TOKENS,
        "reasoning_model": True,
    },
    "mistral-small-24b": {
        "kind": "local",
        "model": "mistral-small-24b-q5km",
        "label": "Mistral-Small-3.2-24B",
        "base_url": OLLAMA_BASE_URL,
        "api_key": "ollama",
        "response_format": "off",
        # A dense 24B leaves most of the card free for a larger context.
        "context_tokens": 32_768,
        "timeout_seconds": 600.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "gguf_file": "Mistral-Small-3.2-24B-Instruct-2506-Q5_K_M.gguf",
        "gguf_sha256": "423446013d1d9a63eb09d84f9d4fe0a1a81b57b6530260afd07a070b4e74845a",
        "quantisation": "Q5_K_M",
        "disk_gb": 16.8,
        "max_tokens": REASONING_MAX_TOKENS,
        "reasoning_model": False,
    },
    "llama33-70b": {
        "kind": "local",
        "model": "llama33-70b-q4km",
        "label": "Llama 3.3 70B Instruct",
        "base_url": OLLAMA_BASE_URL,
        "api_key": "ollama",
        "response_format": "off",
        "context_tokens": 12_288,
        "timeout_seconds": 900.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "gguf_file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "gguf_sha256": "d7f4f209208f984fe95af82d014171f9d05427029746b1ae08297f102a6ddee4",
        "quantisation": "Q4_K_M",
        "disk_gb": 42.52,
        "max_tokens": REASONING_MAX_TOKENS,
        "reasoning_model": False,
    },
}

# Suggested execution order: API models first, then local models smallest
# to largest so the tightest fit is attempted last.
RECOMMENDED_ORDER = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "glm47-flash",
    "qwen36-35b-a3b",
    "mistral-small-24b",
    "llama33-70b",
]

REQUIRED_FIELDS = (
    "kind", "model", "label", "context_tokens", "timeout_seconds",
    "input_cost_per_million", "output_cost_per_million", "max_tokens",
)


def model_keys(include_smoke=False):
    """Registry keys in the recommended execution order.

    Smoke-test entries are excluded unless asked for: they exist to prove the
    plumbing, and including them in a study loop would contaminate results.
    """
    keys = [key for key in RECOMMENDED_ORDER if key in MODEL_REGISTRY]
    extra = [
        key for key in MODEL_REGISTRY
        if key not in RECOMMENDED_ORDER
        and (include_smoke or not MODEL_REGISTRY[key].get("smoke_test_only"))
    ]
    return keys + extra


def get_model(key):
    """Return a copy of one registry entry, validated."""
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key {key!r}. Known keys: {', '.join(model_keys())}"
        )
    entry = dict(MODEL_REGISTRY[key])
    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(
            f"Model {key!r} is missing required field(s): {', '.join(missing)}")
    if entry["kind"] not in {"api", "local"}:
        raise ValueError(
            f"Model {key!r} has unknown kind {entry['kind']!r}; "
            "expected 'api' or 'local'")
    if entry["kind"] == "local" and not entry.get("base_url"):
        raise ValueError(
            f"Local model {key!r} must declare a base_url to reach its server")
    entry["key"] = key
    return entry


def provenance(key):
    """The identity fields a run manifest must record for this model.

    A hosted model is pinned by the dated snapshot the provider reports; a
    local model has no such identifier, so its weights file and hash stand in.
    """
    entry = get_model(key)
    record = {
        "model_key": key,
        "kind": entry["kind"],
        "model": entry["model"],
        "label": entry["label"],
        "context_tokens": entry["context_tokens"],
    }
    if entry["kind"] == "api":
        record["expected_resolved_model"] = entry.get("expected_resolved_model")
        record["base_url"] = entry.get("base_url")
    else:
        record["base_url"] = entry["base_url"]
        record["gguf_file"] = entry.get("gguf_file")
        record["gguf_sha256"] = entry.get("gguf_sha256")
        record["quantisation"] = entry.get("quantisation")
    return record
