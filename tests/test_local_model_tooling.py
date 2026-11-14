"""Guards on the local-weights download and import tooling.

These scripts move 90 GB and then decide which bytes a benchmark run cites as
its model identity, so the parts that can silently go wrong are pinned here:
the source roster, the context length baked into each Ollama model, and the
hash rewrite into the registry.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(script_name, module_name):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FETCH = _load("fetch_local_models.py", "fetch_local_models")
IMPORT = _load("import_local_models.py", "import_local_models")

from configs.model_registry import get_model  # noqa: E402


def test_every_source_matches_a_registry_entry():
    for key, source in FETCH.SOURCES.items():
        entry = get_model(key)
        assert entry["gguf_file"] == source["file"], (
            f"{key}: the file downloaded must be the file the registry names")


def test_downloads_run_smallest_first():
    """A failure should surface after 21 GB, not after 42 GB."""
    sizes = [FETCH.SOURCES[key]["expected_gb"] for key in FETCH.ORDER]
    assert sizes == sorted(sizes)
    assert set(FETCH.ORDER) == set(FETCH.SOURCES)


def test_the_multi_token_prediction_repo_is_not_used_for_qwen():
    """A same-named file exists in the MTP repo at a different size."""
    assert FETCH.SOURCES["qwen36-35b-a3b"]["repo"].endswith("Qwen3.6-35B-A3B-GGUF")
    assert "MTP" not in FETCH.SOURCES["qwen36-35b-a3b"]["repo"]


def test_modelfile_pins_the_rosters_context_length():
    for key in IMPORT.LOCAL_KEYS:
        entry = get_model(key)
        modelfile = IMPORT.build_modelfile("/w.gguf", entry["context_tokens"])
        assert f"PARAMETER num_ctx {entry['context_tokens']}" in modelfile
        assert modelfile.startswith("FROM /w.gguf")


def test_modelfile_does_not_pin_sampling():
    """The planner sets temperature per request; baking it in would override
    the experiment silently."""
    modelfile = IMPORT.build_modelfile("/w.gguf", 4096)
    lowered = modelfile.lower()
    for forbidden in ("temperature", "top_p", "top_k", "seed"):
        assert forbidden not in lowered


def test_llama_modelfile_keeps_the_small_context():
    """42.5 GB of weights on a 48 GB card leaves only the KV cache to shrink."""
    entry = get_model("llama33-70b")
    assert entry["context_tokens"] == 12288
    assert "num_ctx 12288" in IMPORT.build_modelfile("/w.gguf",
                                                    entry["context_tokens"])


def test_hash_rewrite_targets_only_the_named_model(tmp_path, monkeypatch):
    registry = tmp_path / "model_registry.py"
    registry.write_text(
        'MODEL_REGISTRY = {\n'
        '    "glm47-flash": {\n'
        '        "gguf_sha256": None,\n'
        '    },\n'
        '    "llama33-70b": {\n'
        '        "gguf_sha256": None,\n'
        '    },\n'
        '}\n'
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps([
        {"model_key": "glm47-flash", "sha256": "a" * 64},
    ]))
    monkeypatch.setattr(IMPORT, "REGISTRY_FILE", registry)

    assert IMPORT.update_registry_hashes(provenance) == 1
    text = registry.read_text()
    assert f'"gguf_sha256": "{"a" * 64}"' in text
    assert text.count('"gguf_sha256": None') == 1, (
        "only the model named in the provenance file may be rewritten")


def test_missing_provenance_is_reported_not_fatal(tmp_path):
    assert IMPORT.update_registry_hashes(tmp_path / "absent.json") == 0


RUNNER = _load("run_multi_model_benchmark.py", "run_multi_model_benchmark")


def _built_command(tmp_path, **overrides):
    """The argv run_model() hands to generate_results.py, without running it."""
    captured = {}

    class _Args:
        methods = ["ForeSight"]
        scenarios = ["A1_easy_few"]
        trials = 1
        max_llm_calls = 10
        seed = 1
        dry_run = False
        resume = False
        force_rerun = False

    args = _Args()
    args.out_dir = tmp_path
    for name, value in overrides.items():
        setattr(args, name, value)

    def _fake_run(command, env=None):
        captured["command"] = command

        class _Result:
            returncode = 0
        return _Result()

    original = RUNNER.subprocess.run
    RUNNER.subprocess.run = _fake_run
    try:
        RUNNER.run_model("gpt-4o-mini", args, binary=None)
    finally:
        RUNNER.subprocess.run = original
    return captured["command"]


def test_resume_reaches_the_child_process(tmp_path):
    """--resume must pass a directory down, not merely gate --force-rerun.

    It previously did the latter, so an interrupted multi-day run silently
    restarted from zero instead of reusing completed trials.
    """
    command = _built_command(tmp_path, resume=True)
    assert "--resume" in command
    # generate_results.py refuses a resume directory that differs from
    # --out-dir, so the two must name the same per-model directory.
    resume_dir = command[command.index("--resume") + 1]
    out_dir = command[command.index("--out-dir") + 1]
    assert resume_dir == out_dir


def test_force_rerun_does_not_require_resume(tmp_path):
    command = _built_command(tmp_path, force_rerun=True)
    assert "--force-rerun" in command


def test_plain_run_requests_neither(tmp_path):
    command = _built_command(tmp_path)
    assert "--resume" not in command
    assert "--force-rerun" not in command
