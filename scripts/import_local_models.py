#!/usr/bin/env python3
"""Import downloaded GGUF weights into Ollama and record their provenance.

Each model is created with the context length the roster pins for it, which
bounds the KV cache: the 70B at Q4_K_M occupies 42.5 GB of a 48 GB card.

After import the script writes each weights file's SHA-256 back into
``configs/model_registry.py``. A locally served model has no dated snapshot to
pin a run against, so the hash of the file that produced the answers is the
provenance a reader needs.

Usage::

    scripts/import_local_models.py --list
    scripts/import_local_models.py --models glm47-flash
    scripts/import_local_models.py
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.model_registry import MODEL_REGISTRY, get_model  # noqa: E402

DEFAULT_SRC = Path.home() / "models" / "gguf"
REGISTRY_FILE = ROOT / "configs" / "model_registry.py"
LOCAL_KEYS = ["glm47-flash", "qwen36-35b-a3b", "mistral-small-24b", "llama33-70b"]


def ollama_binary():
    found = shutil.which("ollama")
    if found:
        return found
    local = Path.home() / ".local" / "ollama" / "bin" / "ollama"
    if local.exists():
        return str(local)
    raise RuntimeError(
        "ollama not found on PATH or at ~/.local/ollama/bin/ollama")


def server_is_up(binary):
    result = subprocess.run([binary, "list"], capture_output=True, text=True)
    return result.returncode == 0


def build_modelfile(gguf_path, context_tokens):
    """A Modelfile pinning the weights and the context length.

    Sampling is left at the server default; the planner sets temperature per
    request, and overriding it here would silently override the experiment.
    """
    return (
        f"FROM {gguf_path}\n"
        f"PARAMETER num_ctx {context_tokens}\n"
    )


def import_one(key, src_dir, binary, force=False):
    entry = get_model(key)
    tag = entry["model"]
    gguf = src_dir / entry["gguf_file"]
    record = {"model_key": key, "tag": tag, "ok": False, "error": None}

    if not gguf.exists():
        record["error"] = f"weights not found: {gguf}"
        return record

    existing = subprocess.run([binary, "list"], capture_output=True, text=True)
    if not force and tag in (existing.stdout or ""):
        print(f"  already imported as {tag} (use --force to rebuild)")
        record["ok"] = True
        record["skipped"] = True
        return record

    modelfile = build_modelfile(gguf, entry["context_tokens"])
    print(f"  creating {tag} with num_ctx={entry['context_tokens']:,}")
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile",
                                     delete=False) as handle:
        handle.write(modelfile)
        modelfile_path = handle.name
    try:
        result = subprocess.run(
            [binary, "create", tag, "-f", modelfile_path],
            capture_output=True, text=True)
        if result.returncode != 0:
            record["error"] = (
                result.stderr or result.stdout or "ollama create failed"
            ).strip()[:400]
            return record
    finally:
        Path(modelfile_path).unlink(missing_ok=True)

    record["ok"] = True
    return record


def update_registry_hashes(provenance_path):
    """Write each weights file's SHA-256 into the registry as its identity."""
    if not provenance_path.exists():
        print(f"no provenance file at {provenance_path}; hashes not updated")
        return 0
    records = {
        item["model_key"]: item
        for item in json.loads(provenance_path.read_text())
        if item.get("sha256")
    }
    if not records:
        print("provenance file carries no hashes; nothing to update")
        return 0

    source = REGISTRY_FILE.read_text()
    updated = 0
    for key, item in records.items():
        if key not in MODEL_REGISTRY:
            continue
        # Replace the gguf_sha256 line inside this model's block only.
        pattern = re.compile(
            r'("' + re.escape(key) + r'"\s*:\s*\{.*?"gguf_sha256"\s*:\s*)'
            r'(None|"[0-9a-f]*")',
            re.DOTALL)
        new_source, count = pattern.subn(
            lambda m: m.group(1) + f'"{item["sha256"]}"', source, count=1)
        if count:
            source = new_source
            updated += 1
    if updated:
        REGISTRY_FILE.write_text(source)
    print(f"recorded {updated} weights hash(es) into {REGISTRY_FILE.name}")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"Subset of: {', '.join(LOCAL_KEYS)}")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the tag already exists")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    keys = args.models or LOCAL_KEYS
    unknown = [key for key in keys if key not in LOCAL_KEYS]
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")

    if args.list:
        for key in keys:
            entry = get_model(key)
            gguf = args.src / entry["gguf_file"]
            state = "present" if gguf.exists() else "MISSING"
            print(f"{key:16s} -> {entry['model']:22s} "
                  f"ctx={entry['context_tokens']:>6,}  {state}")
        return 0

    binary = ollama_binary()
    if not server_is_up(binary):
        print("Ollama server is not responding. Start it with:\n"
              "  setsid nohup ~/.local/ollama/bin/ollama serve "
              "> ~/.local/ollama/serve.log 2>&1 &")
        return 1

    results = []
    for key in keys:
        print(f"=== {key}", flush=True)
        record = import_one(key, args.src, binary, force=args.force)
        results.append(record)
        print("  OK" if record["ok"] else f"  FAILED: {record['error']}",
              flush=True)

    update_registry_hashes(args.src / "provenance.json")

    failed = [r["model_key"] for r in results if not r["ok"]]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nImported. Validate next:\n"
          "  python3 scripts/validate_models.py --models "
          + " ".join(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
