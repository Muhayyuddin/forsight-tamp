#!/usr/bin/env python3
"""Run the benchmark across every model, one resident model at a time.

Sequential execution is forced, not a preference. The 70B at Q4_K_M occupies
42.5 GB of the 47.1 GiB this card reports, so no two local models fit at once.
Attempting to overlap them evicts weights mid-run, and reloading a 42 GB model
from disk costs far more than it saves.

Per model the loop is: load, run the whole matrix, unload, verify the card is
free, move on. Two details matter for throughput:

* **Keep the model resident during its own run.** Ollama unloads after an idle
  period by default. A planning trial has gaps between requests, and paying a
  42 GB reload in the middle of a scenario would dominate the measurement.
* **Unload explicitly at the end**, rather than waiting for the idle timer, so
  the next model starts against a known-free card.

Results go to ``<out-dir>/<model_key>/`` so runs never collide and the
cross-model analysis can merge them.

Usage::

    scripts/run_multi_model_benchmark.py --dry-run
    scripts/run_multi_model_benchmark.py --models gpt-4o-mini --trials 1
    scripts/run_multi_model_benchmark.py --out-dir results/multimodel
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.model_registry import get_model, model_keys, provenance  # noqa: E402

BENCHMARK_METHODS = [
    "LLM3-FS", "OWL-TAMP", "ViLaIn-TAMP", "VLM-TAMP", "ForeSight",
]
SCENARIOS = [
    "A1_easy_few", "A2_medium_mixed", "PKG5_six_exact_ycb_cans",
    "PUSH1_push_base_stack_cube", "PUSH2_blocked_detour_push_stack",
    "SHF2_medium_shelf", "STK2_three_stack", "STK4_pyramid",
    "TOH1_hanoi_3disk", "TOH4_4DiskCylinder",
]

# Long enough that a model stays resident across the gaps inside a trial, but
# still bounded so a crashed run does not pin the card indefinitely.
RESIDENT_KEEP_ALIVE = "4h"


def ollama_binary():
    found = shutil.which("ollama")
    if found:
        return found
    local = Path.home() / ".local" / "ollama" / "bin" / "ollama"
    if local.exists():
        return str(local)
    return None


def resident_models(binary):
    """Model names currently holding VRAM."""
    if not binary:
        return []
    result = subprocess.run([binary, "ps"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    lines = (result.stdout or "").strip().splitlines()[1:]
    return [line.split()[0] for line in lines if line.strip()]


def unload(binary, tag, wait_seconds=60):
    """Evict a model and wait until the card actually reports it gone."""
    if not binary:
        return True
    subprocess.run([binary, "stop", tag], capture_output=True, text=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if tag not in resident_models(binary):
            return True
        time.sleep(2)
    return False


def unload_all(binary):
    for tag in resident_models(binary):
        unload(binary, tag)


def run_model(key, args, binary):
    entry = get_model(key)
    out_dir = args.out_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "model_key": key,
        "provenance": provenance(key),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
    }

    if entry["kind"] == "local":
        # Start from a known-free card: a model left over from a previous run
        # would otherwise contend for VRAM with this one.
        unload_all(binary)

    command = [
        sys.executable, str(ROOT / "generate_results.py"),
        "--model-key", key,
        "--methods", *args.methods,
        "--scenarios", *args.scenarios,
        "--trials", str(args.trials),
        "--max-llm-calls", str(args.max_llm_calls),
        "--seed", str(args.seed),
        "--out-dir", str(out_dir),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        # generate_results.py resumes from a directory and requires it to match
        # --out-dir, so the per-model directory is both.
        command += ["--resume", str(out_dir)]
    if args.force_rerun:
        command.append("--force-rerun")

    environment = dict(os.environ)
    if entry["kind"] == "local":
        environment["OLLAMA_KEEP_ALIVE"] = RESIDENT_KEEP_ALIVE

    print(f"  $ {' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, env=environment)
    record["elapsed_minutes"] = round((time.time() - started) / 60, 1)
    record["exit_code"] = result.returncode
    record["ok"] = result.returncode == 0

    if entry["kind"] == "local":
        freed = unload(binary, entry["model"])
        record["unloaded"] = freed
        if not freed:
            print(f"  WARNING: {entry['model']} still resident after stop",
                  flush=True)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"Default, in order: {', '.join(model_keys())}")
    parser.add_argument("--methods", nargs="*", default=BENCHMARK_METHODS)
    parser.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "multimodel")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing per-model directories")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Keep going if one model fails")
    args = parser.parse_args()

    keys = args.models or model_keys()
    unknown = [key for key in keys if key not in model_keys(include_smoke=True)]
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")

    binary = ollama_binary()
    if any(get_model(key)["kind"] == "local" for key in keys) and not binary:
        parser.error("local models requested but ollama was not found")

    cells = len(args.methods) * len(args.scenarios) * args.trials
    print(f"{len(keys)} model(s) x {len(args.methods)} methods x "
          f"{len(args.scenarios)} scenarios x {args.trials} trials")
    print(f"= {cells} trials per model, {cells * len(keys)} total")
    print(f"order: {' -> '.join(keys)}")
    print(f"out:   {args.out_dir}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, key in enumerate(keys, start=1):
        entry = get_model(key)
        print(f"=== [{index}/{len(keys)}] {entry['label']} ({key})",
              flush=True)
        record = run_model(key, args, binary)
        records.append(record)
        status = "OK" if record["ok"] else f"FAILED ({record['exit_code']})"
        print(f"  {status} in {record['elapsed_minutes']} min\n", flush=True)

        summary = args.out_dir / "multimodel_run_log.json"
        summary.write_text(json.dumps(records, indent=2))

        if not record["ok"] and not args.continue_on_failure:
            print("Stopping: pass --continue-on-failure to carry on.")
            return 1

    failed = [r["model_key"] for r in records if not r["ok"]]
    total = sum(r["elapsed_minutes"] for r in records)
    print(f"total wall clock: {total / 60:.1f} h")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All models completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
