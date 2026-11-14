#!/usr/bin/env python3
"""Record an MP4 and a GIF of ForeSight executing each scenario in the paper.

Scenarios run in sequence, each with --close-when-done so the batch does not
stop in a replay loop. The video is rendered headless from the final attempt's
committed execution.

These are illustrations, not measurements. Each run is a fresh planning episode
with its own language-model calls and can fail, so each scenario is retried
until the goal is reached. recordings.json next to the videos says, for each
scenario, whether the clip that survived shows a solved run.

Usage::

    python3 scripts/record_paper_scenarios.py                  # all ten
    python3 scripts/record_paper_scenarios.py --dry-run
    python3 scripts/record_paper_scenarios.py --scenarios TOH4_4DiskCylinder
    python3 scripts/record_paper_scenarios.py --skip-existing  # resume a batch
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def paper_scenarios():
    """The ten scenarios the paper reports, read from the benchmark runner.

    Loaded by path rather than ``import scripts.…``: ROS installs a package
    named ``scripts`` that shadows this directory, so the package import
    resolves to ROS and fails.
    """
    source = ROOT / "scripts" / "run_multi_model_benchmark.py"
    spec = importlib.util.spec_from_file_location("_fs_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.SCENARIOS)


def command(scenario, args, out_dir):
    # The video is rendered from the final attempt's committed execution, not
    # recorded from the window, so it shows no planning and runs headless.
    cmd = [
        sys.executable, str(ROOT / "run_foresight.py"),
        "--scenario", scenario,
        "--model", args.model,
        "--max-llm-calls", str(args.max_llm_calls),
        "--play-traj",
        "--no-gui",
        "--render-video", str(out_dir),
        "--video-size", args.video_size,
        "--save-gif",
        "--close-when-done",
        "--require-success",
    ]
    if args.llm_base_url:
        cmd += ["--llm-base-url", args.llm_base_url,
                "--llm-api-key", args.llm_api_key,
                "--llm-response-format", "off"]
    # Stills need the GUI, so they are opt-in and run as a GUI pass.
    if args.with_stills:
        cmd = [part for part in cmd if part != "--no-gui"]
        cmd += ["--stills", str(args.stills_dir),
                "--still-count", str(args.still_count),
                "--still-size", args.still_size]
    return cmd


def outputs(out_dir, scenario):
    return out_dir / f"{scenario}.mp4", out_dir / f"{scenario}.gif"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "videos")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Defaults to the paper's ten.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-llm-calls", type=int, default=20)
    parser.add_argument("--llm-base-url", default=None,
                        help="OpenAI-compatible endpoint, e.g. Ollama's.")
    parser.add_argument("--llm-api-key", default="ollama")
    parser.add_argument("--video-size", default="1920x1080",
                        help="Pixel size of each rendered video.")
    parser.add_argument("--with-stills", action="store_true",
                        help="Also render still frames of each run (needs the GUI).")
    parser.add_argument("--stills-dir", type=Path,
                        default=ROOT / "videos" / "stills",
                        help="Where the high-resolution PNGs are written.")
    parser.add_argument("--still-count", type=int, default=3,
                        help="Stills per scenario; the filmstrip uses three.")
    parser.add_argument("--still-size", default="1600x1200",
                        help="Pixel size of each still.")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Seconds before a scenario is abandoned.")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Tries per scenario before keeping a failed run.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Leave scenarios already recorded as solved.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = args.scenarios or paper_scenarios()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "recordings.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            manifest = {}

    print(f"recording {len(scenarios)} scenarios -> {out_dir}")
    results = []
    for index, scenario in enumerate(scenarios, start=1):
        mp4, gif = outputs(out_dir, scenario)
        recorded = manifest.get(scenario, {})
        if (args.skip_existing and recorded.get("solved")
                and mp4.exists() and gif.exists()):
            print(f"[{index}/{len(scenarios)}] {scenario}: already solved")
            results.append((scenario, "kept, solved", 0.0, 0))
            continue
        print(f"[{index}/{len(scenarios)}] {scenario}", flush=True)
        if args.dry_run:
            print("  $ " + " ".join(command(scenario, args, out_dir)))
            results.append((scenario, "dry-run", 0.0, 0))
            continue

        started = time.monotonic()
        solved, attempt, status = False, 0, "not run"
        # Each attempt is a fresh episode, not a retry of the same one, so a
        # scenario that failed on its own merits can still come good.
        while attempt < max(1, args.attempts) and not solved:
            attempt += 1
            try:
                code = subprocess.run(command(scenario, args, out_dir),
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                status = "timed out"
                continue
            solved = code == 0
            status = "solved" if solved else f"goal not reached (exit {code})"
            if not solved and attempt < args.attempts:
                print(f"  attempt {attempt}: {status}; retrying", flush=True)
        elapsed = time.monotonic() - started

        if mp4.exists() and gif.exists():
            files = "mp4 + gif"
        elif mp4.exists():
            files = "mp4 only"
        else:
            files = "no video"
        detail = f"{status}, {files}"
        results.append((scenario, detail, elapsed, attempt))
        manifest[scenario] = {
            "solved": solved,
            "attempts": attempt,
            "mp4": mp4.name if mp4.exists() else None,
            "gif": gif.name if gif.exists() else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2,
                                            sort_keys=True) + "\n")
        print(f"  {detail} after {attempt} attempt(s), "
              f"{elapsed / 60:.1f} min", flush=True)

    print("\n--- summary ---")
    width = max(len(name) for name, _, _, _ in results)
    for scenario, detail, elapsed, attempt in results:
        print(f"  {scenario:<{width}}  {detail:<34} "
              f"{attempt} try  {elapsed / 60:5.1f} min")
    usable = [name for name, detail, _, _ in results
              if "solved" in detail and "mp4 + gif" in detail
              or detail == "kept, solved"]
    print(f"\n{len(usable)}/{len(results)} are recordings of a solved run")
    if len(usable) != len(results):
        missing = [name for name, _, _, _ in results if name not in usable]
        print("still unsolved: " + ", ".join(missing))
        print("re-run just those with --scenarios " + " ".join(missing))
    print(f"files in {out_dir}; manifest at {manifest_path.name}")
    return 0 if len(usable) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
