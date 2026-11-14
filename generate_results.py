#!/usr/bin/env python3
"""
generate_results.py: run planning methods on the ForeSight-TAMP scenarios.

Runs every selected method on every selected scenario for N trials, prints a
per-scenario summary, and writes the results to an output directory:

    raw_trials.json       per-trial records (used to resume an interrupted run)
    results.json          per-method, per-scenario summaries
    results_summary.csv   the same summaries as a flat CSV
    run_config.json       the settings needed to reproduce the run
    traces/               per-trial event traces and LLM payloads

Usage:
    python3 generate_results.py --model-key gpt-4o-mini --trials 3
    python3 generate_results.py --methods ForeSight --scenarios A2 SHF2
    python3 generate_results.py --groups PACK PUSH --trials 1
    python3 generate_results.py --methods Abl_base Abl_acfm_only --trials 3
    python3 generate_results.py --dry-run            # mock data, no LLM calls

Scenario groups:
    PACK → packing          (A1, A2, PKG5)
    PUSH → push-stack       (PUSH1, PUSH2)
    SHF  → shelf placement  (SHF2)
    STK  → stacking         (STK2, STK4)
    TOH  → Tower of Hanoi   (TOH1, TOH4)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_results")

# ─────────────────────────── PATHS ────────────────────────────
ROOT = Path(__file__).parent
INSTANCE_DIR = ROOT / "envs" / "task_instances" / "foresight"
RESULTS_BASE = ROOT / "results"

# ─────────────────────────── METHODS ──────────────────────────
# Runnable methods. ``BENCHMARK_METHODS`` below is the controlled comparison;
# the remaining entries are ablations and controls.
# ``class`` maps to the runner used by ``run_trial``.
METHODS = {
    "LLM3-FS":     {"class": "llm3",       "variant": "from_scratch", "label": r"LLM\textsuperscript{3}-FS"},
    "OWL-TAMP":    {"class": "owl_tamp",   "label": "OWL-TAMP"},
    "ViLaIn-TAMP": {"class": "vilain_tamp","label": "ViLaIn"},
    "VLM-TAMP":    {"class": "vlm_tamp",   "label": "VLM-TAMP"},
    # ForeSight full + ablations
    "ForeSight":          {"class": "foresight", "disable": [],                    "label": r"\textbf{ForeSight}"},
    "Abl_base": {
        "class": "foresight",
        "disable": ["acfm", "rehearsal", "cascade"],
        "label": "Base planner",
    },
    "Abl_acfm_only": {
        "class": "foresight",
        "disable": ["rehearsal", "cascade"],
        "label": "ACFM only",
    },
    "Abl_rehearsal_only": {
        "class": "foresight",
        "disable": ["acfm", "cascade"],
        "label": "Rehearsal only",
    },
    "Abl_acfm_rehearsal": {
        "class": "foresight",
        "disable": ["cascade"],
        "label": r"ACFM + Rehearsal",
    },
    "Abl_no_ranking": {
        "class": "foresight",
        # Hard masks, per-step binding and candidate rollout all stay; only
        # the soft ordering among feasible cells is removed. Separates
        # ranking geometry from filtering it, which the coarse "acfm" flag
        # cannot do because it also swaps in the one-shot planner.
        "disable": ["ranking"],
        "label": "w/o ACFM ranking",
    },
    "Abl_direct_top1": {
        "class": "foresight",
        "disable": [],
        "candidate_selection_mode": "top1",
        "label": r"Direct ACFM top-1 (no LLM)",
    },
    "Abl_matched_rollout": {
        "class": "foresight",
        "disable": [],
        "candidate_selection_mode": "rollout_control",
        # Must stay equal to DEFAULT_ROLLOUT_VALIDATION_QUOTAS: this arm exists
        # to give the non-LLM control the same simulator budget ForeSight gets,
        # so any divergence would hand one side more compute than the other.
        "rollout_validation_quotas": {
            "place": 10,
            "push": 9,
            "stacking_place": 1,
            "hanoi_place": 2,
        },
        "label": r"Matched-rollout-budget control",
    },
}

# The five methods of the controlled comparison.
BENCHMARK_METHODS = [
    "LLM3-FS",
    "OWL-TAMP",
    "ViLaIn-TAMP",
    "VLM-TAMP",
    "ForeSight",
]
DEFAULT_METHODS = list(BENCHMARK_METHODS)

# The ten evaluated scenarios.
PAPER_SCENARIOS = [
    "A1_easy_few",
    "A2_medium_mixed",
    "PKG5_six_exact_ycb_cans",
    "PUSH1_push_base_stack_cube",
    "PUSH2_blocked_detour_push_stack",
    "SHF2_medium_shelf",
    "STK2_three_stack",
    "STK4_pyramid",
    "TOH1_hanoi_3disk",
    "TOH4_4DiskCylinder",
]

# The factorial ablation and its controls.
ABLATION_METHODS = [
    "Abl_base",
    "Abl_acfm_only",
    "Abl_rehearsal_only",
    "Abl_acfm_rehearsal",
    "Abl_no_ranking",
    "Abl_direct_top1",
    "Abl_matched_rollout",
]


def protocol_method_config(method_name: str) -> dict:
    """Return the execution-relevant portion of a method configuration."""
    config = METHODS[method_name]
    fields = (
        "class",
        "variant",
        "disable",
        "candidate_selection_mode",
        "rollout_validation_quotas",
    )
    return {
        field: config[field]
        for field in fields
        if field in config
    }

# Scenario groups by task family.
GROUP_SCENARIOS = {
    "PACK": ["A1", "A2", "PKG5"],
    "PUSH": ["PUSH1", "PUSH2"],
    "SHF": ["SHF2"],
    "STK": ["STK2", "STK4"],
    "TOH": ["TOH1", "TOH4"],
}

GROUP_LABELS = {
    "PACK": "Packing",
    "PUSH": "Push-stack",
    "SHF": "Shelf placement",
    "STK": "Stacking",
    "TOH": "Tower of Hanoi",
}


def all_group_sids(groups: list[str] | None = None) -> list[str]:
    """Return scenario prefixes for the requested existing groups."""
    keys = groups or list(GROUP_SCENARIOS)
    return [
        sid
        for group in keys
        for sid in GROUP_SCENARIOS.get(group, [])
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    temporary.replace(path)


def _directory_hashes(directory: Path) -> dict[str, str]:
    """Return a stable manifest for every regular artifact file."""
    if not directory.is_dir():
        return {}
    root = directory.resolve()
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact trees may not contain symlinks: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Artifact path escapes its trial directory: {path}")
        hashes[path.relative_to(directory).as_posix()] = _sha256_file(path)
    return hashes


def _install_run_log(out_dir: Path) -> Path:
    """Persist all Python logging for a fresh run or compatible resume."""
    path = (out_dir / "run.log").resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", "")).resolve()
            if existing == path:
                return path
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root_logger.addHandler(handler)
    return path


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def _benchmark_input_files() -> list[Path]:
    """Enumerate runtime/protocol inputs, including untracked source files."""
    excluded_roots = {".git", "venv", "results", "docs", "__pycache__"}
    paths = []
    for directory, child_directories, filenames in os.walk(ROOT):
        child_directories[:] = [
            name for name in child_directories if name not in excluded_roots
        ]
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            relative = path.relative_to(ROOT)
            relative_text = relative.as_posix()
            if (
                path.suffix == ".py"
                or relative_text.startswith("prompts/")
                or relative_text.startswith("envs/task_instances/foresight/")
                or relative_text == "requirements.txt"
            ):
                paths.append(path)
    return sorted(paths)


def _benchmark_input_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in _benchmark_input_files()
    }


def _code_provenance() -> dict:
    status = _git_output(
        "status",
        "--porcelain=v1",
        "--",
        "*.py",
        "prompts",
        "envs/task_instances/foresight",
    )
    diff = _git_output(
        "diff",
        "--no-ext-diff",
        "HEAD",
        "--",
        "*.py",
        "prompts",
        "envs/task_instances/foresight",
    )
    input_hashes = _benchmark_input_hashes()
    input_tree_payload = json.dumps(
        input_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "git_commit": _git_output("rev-parse", "HEAD").strip() or None,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "relevant_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "benchmark_input_count": len(input_hashes),
        "benchmark_input_tree_sha256": hashlib.sha256(
            input_tree_payload
        ).hexdigest(),
    }


def _dependency_snapshot() -> dict:
    packages = sorted(
        {
            distribution.metadata.get("Name", distribution.name): distribution.version
            for distribution in importlib.metadata.distributions()
        }.items(),
        key=lambda item: item[0].lower(),
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {name: version for name, version in packages},
    }


def _archive_protocol_inputs(out_dir: Path, scenario_stems: list[str]) -> dict:
    """Copy immutable prompt/scenario inputs and record the dirty code state."""
    snapshot_dir = out_dir / "protocol_snapshot"
    prompt_dir = snapshot_dir / "prompts"
    scenario_dir = snapshot_dir / "scenarios"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    source_dir = snapshot_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    hashes = _benchmark_input_hashes()
    for source in _benchmark_input_files():
        relative = source.relative_to(ROOT)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in sorted((ROOT / "prompts").rglob("*.txt")):
        relative = source.relative_to(ROOT)
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for stem in scenario_stems:
        source = INSTANCE_DIR / f"{stem}.json"
        if not source.exists():
            source = sorted(INSTANCE_DIR.glob(f"{stem}*.json"))[0]
        destination = scenario_dir / source.name
        shutil.copy2(source, destination)

    status = _git_output(
        "status",
        "--porcelain=v1",
        "--",
        "*.py",
        "prompts",
        "envs/task_instances/foresight",
    )
    relevant_diff = _git_output(
        "diff",
        "--no-ext-diff",
        "HEAD",
        "--",
        "*.py",
        "prompts",
        "envs/task_instances/foresight",
    )
    (snapshot_dir / "git_status.txt").write_text(status, encoding="utf-8")
    (snapshot_dir / "relevant_code.patch").write_text(
        relevant_diff, encoding="utf-8"
    )
    _atomic_write_json(snapshot_dir / "dependencies.json", _dependency_snapshot())
    _atomic_write_json(snapshot_dir / "input_hashes.json", hashes)
    return {
        "directory": str(snapshot_dir.relative_to(out_dir)),
        "input_hashes_sha256": _sha256_file(snapshot_dir / "input_hashes.json"),
        "dependencies_sha256": _sha256_file(snapshot_dir / "dependencies.json"),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "relevant_diff_sha256": hashlib.sha256(
            relevant_diff.encode("utf-8")
        ).hexdigest(),
    }


def _validate_resume_checkpoint(
    raw_trials: dict,
    *,
    out_dir: Path,
    methods: list[str],
    scenarios: list[str],
    trials_per_cell: int,
    max_llm_requests: int,
    base_seed: int,
) -> None:
    """Reject incomplete, corrupted, or cross-protocol resume records."""
    unexpected_methods = sorted(set(raw_trials) - set(methods))
    if unexpected_methods:
        raise ValueError(
            f"raw checkpoint contains unexpected methods: {unexpected_methods}"
        )
    root = out_dir.resolve()
    scenario_indices = {scenario: index for index, scenario in enumerate(scenarios)}
    for method, method_rows in raw_trials.items():
        if not isinstance(method_rows, dict):
            raise ValueError(f"raw checkpoint method {method} is not an object")
        unexpected_scenarios = sorted(set(method_rows) - set(scenarios))
        if unexpected_scenarios:
            raise ValueError(
                f"raw checkpoint contains unexpected {method} scenarios: "
                f"{unexpected_scenarios}"
            )
        for scenario, rows in method_rows.items():
            if not isinstance(rows, list) or len(rows) > trials_per_cell:
                raise ValueError(
                    f"invalid trial list for {method} x {scenario}: "
                    f"expected at most {trials_per_cell} rows"
                )
            for trial_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"trial {method} x {scenario} x {trial_index} is not an object"
                    )
                expected_seed = (
                    base_seed
                    + scenario_indices[scenario] * trials_per_cell
                    + trial_index
                )
                if row.get("trial_index") != trial_index:
                    raise ValueError(
                        f"trial index mismatch for {method} x {scenario}: "
                        f"expected {trial_index}, observed {row.get('trial_index')}"
                    )
                if row.get("trial_seed") != expected_seed:
                    raise ValueError(
                        f"trial seed mismatch for {method} x {scenario} x "
                        f"{trial_index}"
                    )
                if int(row.get("llm_api_requests", row.get("num_llm_calls", 0))) > max_llm_requests:
                    raise ValueError(
                        f"LLM request cap exceeded in {method} x {scenario} x "
                        f"{trial_index}"
                    )
                trace_relative = row.get("trace_file")
                trace_hash = row.get("trace_sha256")
                if not trace_relative or not trace_hash:
                    raise ValueError(
                        f"missing trace provenance for {method} x {scenario} x "
                        f"{trial_index}"
                    )
                trace_path = (out_dir / trace_relative).resolve()
                if not trace_path.is_relative_to(root) or not trace_path.is_file():
                    raise ValueError(
                        f"invalid trace path for {method} x {scenario} x {trial_index}"
                    )
                if _sha256_file(trace_path) != trace_hash:
                    raise ValueError(
                        f"trace hash mismatch for {method} x {scenario} x {trial_index}"
                    )
                with trace_path.open(encoding="utf-8") as stream:
                    trace_payload = json.load(stream)
                expected_identity = {
                    "method": method,
                    "scenario": scenario,
                    "trial_index": trial_index,
                    "trial_seed": expected_seed,
                }
                if any(
                    trace_payload.get(key) != value
                    for key, value in expected_identity.items()
                ):
                    raise ValueError(
                        f"trace identity mismatch for {method} x {scenario} x "
                        f"{trial_index}"
                    )
                if METHODS.get(method, {}).get("class") == "foresight":
                    artifact_relative = row.get("artifact_directory")
                    artifact_hashes = row.get("artifact_files_sha256")
                    expected_relative = (
                        Path("artifacts") / method / scenario
                        / f"trial_{trial_index:03d}"
                    )
                    if (
                        artifact_relative != expected_relative.as_posix()
                        or not isinstance(artifact_hashes, dict)
                    ):
                        raise ValueError(
                            f"invalid artifact provenance for {method} x "
                            f"{scenario} x {trial_index}"
                        )
                    artifact_directory = (out_dir / artifact_relative).resolve()
                    if (
                        not artifact_directory.is_relative_to(root)
                        or not artifact_directory.is_dir()
                    ):
                        raise ValueError(
                            f"invalid artifact directory for {method} x "
                            f"{scenario} x {trial_index}"
                        )
                    observed_hashes = _directory_hashes(artifact_directory)
                    if observed_hashes != artifact_hashes:
                        raise ValueError(
                            f"artifact hash mismatch for {method} x {scenario} "
                            f"x {trial_index}"
                        )


# ─────────────────────────── SCENARIO LOADING ─────────────────

def load_scenario(sid: str) -> dict | None:
    """Load a scenario JSON by prefix (e.g. 'A1' matches 'A1_easy_few.json')."""
    matches = list(INSTANCE_DIR.glob(f"{sid}*.json"))
    if not matches:
        logger.warning(f"No scenario file found for prefix '{sid}'")
        return None
    with open(matches[0]) as f:
        return json.load(f)


def load_group(group: str) -> list[tuple[str, dict]]:
    """Return [(sid_short, scenario_dict), ...] for a group prefix."""
    results = []
    for sid_short in GROUP_SCENARIOS[group]:
        s = load_scenario(sid_short)
        if s:
            results.append((sid_short, s))
    return results


def load_selected_scenarios(
    scenario_ids: list[str],
) -> list[tuple[str, dict]]:
    """Load an explicit scenario list, preserving its requested order.

    A selector may be either a complete JSON stem or an unambiguous prefix.
    Results use the complete stem so checkpoint and CSV rows identify the
    exact scenario that was executed.
    """
    results = []
    seen = set()
    for selector in scenario_ids:
        exact_path = INSTANCE_DIR / f"{selector}.json"
        matches = [exact_path] if exact_path.exists() else sorted(
            INSTANCE_DIR.glob(f"{selector}*.json")
        )
        if not matches:
            raise ValueError(f"No scenario file found for '{selector}'")
        if len(matches) > 1:
            names = ", ".join(path.stem for path in matches)
            raise ValueError(
                f"Scenario selector '{selector}' is ambiguous: {names}"
            )
        path = matches[0]
        if path.stem in seen:
            raise ValueError(f"Scenario '{path.stem}' was selected more than once")
        with path.open() as stream:
            results.append((path.stem, json.load(stream)))
        seen.add(path.stem)
    return results


def build_task_config(scenario: dict) -> dict:
    env_type = scenario.get("env_type", "packing")
    audit_metadata = {
        "scenario_id": scenario.get("scenario_id"),
        "split": scenario.get("split", scenario.get("benchmark_split")),
        "domain": scenario.get("domain", env_type),
        "weight_profile": scenario.get("weight_profile"),
        "threshold": scenario.get("threshold"),
        "push_cost_coefficient": scenario.get("push_cost_coefficient"),
    }

    def with_audit_metadata(config):
        config.update({
            key: value for key, value in audit_metadata.items()
            if value is not None
        })
        return config

    if env_type == "stacking":
        return with_audit_metadata({"boxes": scenario["boxes"],
                                    "stacking_order": scenario["stacking_order"],
                                    "stacking_zone": scenario.get("stacking_zone"),
                                    "task": scenario.get("task")})
    elif env_type == "shelf":
        return with_audit_metadata({"boxes": scenario["boxes"],
                                    "shelf_config": scenario["shelf_config"],
                                    "task": scenario.get("task")})
    elif env_type == "hanoi":
        return with_audit_metadata({"pegs": scenario["pegs"],
                                    "disks": scenario["disks"],
                                    "task": scenario.get("task")})
    elif env_type == "push_stacking":
        return with_audit_metadata({"boxes": scenario["boxes"],
                                    "stacking_order": scenario["stacking_order"],
                                    "target_pose": scenario["target_pose"],
                                    "initial_joint_positions": scenario.get(
                                        "initial_joint_positions"),
                                    "approach_tcp_pose": scenario.get("approach_tcp_pose"),
                                    "task": scenario.get("task")})
    else:
        return with_audit_metadata({"basket": scenario["basket"],
                                    "boxes": scenario["boxes"],
                                    "task_type": scenario.get("task_type", "packing"),
                                    "stacking_order": scenario.get("stacking_order"),
                                    "task": scenario.get("task")})

# ─────────────────────────── SINGLE TRIAL ─────────────────────

def _make_env(env_type: str):
    from envs.pack_compact_env import PackCompactEnv
    from envs.stacking_env import StackingEnv
    from envs.shelf_env import ShelfEnv
    from envs.hanoi_env import TowerOfHanoiEnv
    from envs.push_stacking_env import PushStackingEnv
    if env_type == "stacking":
        return StackingEnv()
    elif env_type == "shelf":
        return ShelfEnv()
    elif env_type == "hanoi":
        return TowerOfHanoiEnv()
    elif env_type == "push_stacking":
        return PushStackingEnv()
    else:
        return PackCompactEnv()


def run_trial(
    method_name: str,
    method_cfg: dict,
    scenario: dict,
    max_llm_calls: int,
    use_gui: bool = False,
    play_traj: bool = False,
    artifact_dir: Path | None = None,
) -> dict:
    """Run one trial of a method on a scenario. Returns metrics dict."""
    env_type = scenario.get("env_type", "packing")
    task_config = build_task_config(scenario)
    t0 = time.time()

    env = _make_env(env_type)

    try:
        cls = method_cfg["class"]

        if cls == "llm3":
            from planners.llm_tamp_planner import LLMTAMPPlanner
            variant = method_cfg.get("variant", "backtrack")
            planner_file = "from_scratch" if variant == "from_scratch" else "backtrack"
            env_prompt = {
                "packing": "pack_boxes.txt",
                "stacking": "stacking.txt",
                "shelf": "shelf.txt",
                "hanoi": "hanoi.txt",
                "push_stacking": "push_stacking.txt",
            }[env_type]
            planner = LLMTAMPPlanner(
                planner_prompt_file=planner_file,
                env_desc_file=env_prompt,
                primitive_actions=env.primitive_actions,
                with_mp_feedback=True,
                trace_size=5,
                task_file=task_config.get("task"),
            )
            result = _run_llm3(
                env,
                planner,
                task_config,
                max_llm_calls,
                use_gui=use_gui,
                play_traj=play_traj,
            )

        elif cls == "owl_tamp":
            from baselines.owl_tamp_baseline import OWLTAMPBaseline
            b = OWLTAMPBaseline(
                env.primitive_actions, env, max_llm_calls=max_llm_calls)
            result = b.run_episode(
                task_config, play_traj=play_traj, use_gui=use_gui)

        elif cls == "vilain_tamp":
            from baselines.vilain_tamp_baseline import ViLaInTAMPBaseline
            b = ViLaInTAMPBaseline(
                env.primitive_actions, env, max_llm_calls=max_llm_calls)
            result = b.run_episode(
                task_config, play_traj=play_traj, use_gui=use_gui)

        elif cls == "vlm_tamp":
            from baselines.vlm_tamp_baseline import VLMTAMPBaseline
            b = VLMTAMPBaseline(
                env.primitive_actions, env, max_llm_calls=max_llm_calls)
            result = b.run_episode(
                task_config, play_traj=play_traj, use_gui=use_gui)

        elif cls == "foresight":
            from main_foresight import ForeSightTAMP
            foresight_options = dict(
                env=env,
                max_llm_calls=max_llm_calls,
                disable_modules=method_cfg.get("disable", []),
                use_gui=use_gui,
                candidate_selection_mode=method_cfg.get(
                    "candidate_selection_mode", "llm"),
                rollout_validation_quotas=method_cfg.get(
                    "rollout_validation_quotas"),
                render_acfm_maps=RENDER_ACFM_MAPS,
                export_trajectory_artifacts=EXPORT_TRAJECTORY_ARTIFACTS,
            )
            if artifact_dir is not None:
                foresight_options["save_dir"] = artifact_dir
            foresight = ForeSightTAMP(**foresight_options)
            result = foresight.run_episode(
                task_config, play_traj=play_traj)

        else:
            raise ValueError(f"Unknown class: {cls}")

    except Exception as exc:
        # A runtime error is recorded as an unsuccessful trial, so it is
        # indistinguishable from a planning failure unless the cause is kept.
        # Retain the full traceback in both the log and the trial record.
        error_traceback = traceback.format_exc()
        logger.error(
            f"  Error in {method_name}: {type(exc).__name__}: {exc}\n{error_traceback}"
        )
        result = {
            "goal_achieved": False,
            "num_llm_calls": 0,
            "num_mp_calls": 0,
            "discarded_simulator_primitives": 0,
            "wasted_real_actions": 0,  # alias of discarded_simulator_primitives
            "rehearsal_saves": 0,
            "cascade_depth": 0,
            "acfm_accuracy": 0.0,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "error_traceback": error_traceback,
        }
    finally:
        try:
            env.destroy()
        except Exception:
            pass

    result.setdefault("total_time", round(time.time() - t0, 2))
    result.setdefault("rehearsal_saves", 0)
    result.setdefault("cascade_depth", 0)
    result.setdefault("acfm_accuracy", 0.0)
    result.setdefault(
        "high_score_selection_rate", result.get("acfm_accuracy", 0.0)
    )
    result.setdefault(
        "discarded_simulator_primitives",
        result.get("wasted_real_actions", 0),
    )
    # ``wasted_real_actions`` is an alias of the authoritative
    # ``discarded_simulator_primitives``.
    result["wasted_real_actions"] = result["discarded_simulator_primitives"]
    return result


def _run_llm3(
    env,
    planner,
    task_config,
    max_llm_calls,
    use_gui=False,
    play_traj=False,
):
    """LLM³ planning loop (backtrack / from_scratch share the same loop)."""
    last_feedback_list = []
    last_temp_plan = None
    num_mp_calls = 0
    num_llm_calls = 0
    discarded_simulator_primitives = 0
    committed_successful_primitives = 0
    feedback = None

    for _ in range(max_llm_calls):
        obs, obs_text = env.reset(**{k: v for k, v in task_config.items()
                                     if k != "task"}, use_gui=use_gui)
        plan = planner.plan(obs_text, last_feedback_list,
                            symbolic_plan=env.get_symbolic_plan())
        last_feedback_list = []
        num_llm_calls += 1
        if plan is None:
            continue

        temp_plan = []
        same_as_last = True
        for action_i, action in enumerate(plan):
            if (same_as_last and last_temp_plan is not None
                    and len(last_temp_plan) > action_i):
                if str(action) == str(last_temp_plan[action_i]):
                    action = last_temp_plan[action_i]
                else:
                    same_as_last = False
            if action.traj is None or len(action.traj) == 0:
                num_mp_calls += 1
            _, feedback = env.step(action, play_traj=play_traj)
            last_feedback_list.append((action, feedback))
            if feedback.action_success:
                temp_plan.append(action)
                committed_successful_primitives += 1
            else:
                action.traj = []
                break
            if feedback.goal_achieved:
                break

        last_temp_plan = temp_plan
        if feedback and feedback.goal_achieved:
            break
        # The next reset (or failed episode termination) discards all
        # successful state-changing primitives from this unsuccessful attempt.
        # A motion-planning failure is not counted because no simulator
        # primitive was executed. Rehearsal actions are accounted separately.
        discarded_simulator_primitives += len(temp_plan)

    return {
        "goal_achieved": feedback.goal_achieved if feedback else False,
        "num_llm_calls": num_llm_calls,
        "num_mp_calls": num_mp_calls,
        "committed_successful_simulator_primitives": committed_successful_primitives,
        "discarded_simulator_primitives": discarded_simulator_primitives,
        "wasted_real_actions": discarded_simulator_primitives,
        "rehearsal_saves": 0,
        "cascade_depth": 0,
        "acfm_accuracy": 0.0,
    }

# ─────────────────────────── SUMMARY ──────────────────────────

def summarise_trials(trials: list[dict]) -> dict:
    successes = [t["goal_achieved"] for t in trials]
    success_times = [
        t.get("total_time", 0) for t in trials if t["goal_achieved"]
    ]
    failure_times = [
        t.get("total_time", 0) for t in trials if not t["goal_achieved"]
    ]
    discarded = [
        t.get(
            "discarded_simulator_primitives",
            t.get("wasted_real_actions", 0),
        )
        for t in trials
    ]
    high_score_rates = [
        t.get("high_score_selection_rate", t.get("acfm_accuracy", 0.0))
        for t in trials
    ]
    summary = {
        "TPSR":           round(np.mean(successes) * 100, 1),
        "TPSR_std":       round(np.std(successes) * 100, 1),
        "avg_llm":        round(np.mean([t.get("num_llm_calls", 0)      for t in trials]), 1),
        "avg_mp":         round(np.mean([t.get("num_mp_calls", 0)       for t in trials]), 1),
        "avg_time":       round(np.mean([t.get("total_time", 0)         for t in trials]), 1),
        "avg_success_time": round(np.mean(success_times), 1) if success_times else None,
        "avg_failure_time": round(np.mean(failure_times), 1) if failure_times else None,
        "avg_discarded_simulator_primitives": round(np.mean(discarded), 2),
        "avg_wasted":     round(np.mean(discarded), 2),  # alias of the above
        "avg_rehearsal":  round(np.mean([t.get("rehearsal_saves", 0)    for t in trials]), 2),
        "avg_cascade":    round(np.mean([t.get("cascade_depth", 0)      for t in trials]), 2),
        "avg_high_score_selection_rate": round(np.mean(high_score_rates), 2),
        "avg_acfm_acc":   round(np.mean(high_score_rates), 2),
        "avg_llm_transport_retries": round(np.mean([
            t.get("llm_transport_retries", 0) for t in trials
        ]), 2),
        "avg_llm_truncated": round(np.mean([
            t.get("llm_truncated_responses", 0) for t in trials
        ]), 2),
        "avg_prompt_tokens": round(np.mean([
            t.get("llm_prompt_tokens", 0) for t in trials
        ]), 1),
        "avg_completion_tokens": round(np.mean([
            t.get("llm_completion_tokens", 0) for t in trials
        ]), 1),
        "avg_candidate_rollout_attempts": round(np.mean([
            t.get("candidate_rollout_attempts", 0) for t in trials
        ]), 1),
        "avg_candidate_rollout_passes": round(np.mean([
            t.get("candidate_rollout_passes", 0) for t in trials
        ]), 1),
        "avg_matched_rollout_quota": round(np.mean([
            t.get("matched_rollout_validation_quota", 0) for t in trials
        ]), 1),
        "avg_matched_rollout_attempted": round(np.mean([
            t.get("matched_rollout_validation_attempted", 0) for t in trials
        ]), 1),
        "explicit_errors": sum(1 for t in trials if t.get("error")),
        "n":              len(trials),
    }
    errors = [t.get("error", "") for t in trials if t.get("error")]
    if errors:
        summary["error"] = errors[-1]
    return summary


# ─────────────────────────── MOCK DATA ────────────────────────

def make_mock_results(
    groups: list[str],
    num_trials: int,
    scenario_ids: list[str] | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Return deterministic fake results for --dry-run mode."""
    rng = np.random.default_rng(42)
    results = {}
    for m in methods or list(METHODS):
        cfg = METHODS[m]
        results[m] = {}
        selected_ids = scenario_ids or all_group_sids(groups)
        for sid_short in selected_ids:
            is_fs = cfg["class"] == "foresight"
            disable = cfg.get("disable", [])
            # Mock TPSR: ForeSight > baselines, ablations in between
            base_tpsr = 0.65 if (is_fs and not disable) else \
                         0.52 if (is_fs and "rehearsal" not in disable) else \
                         0.45 if (is_fs) else \
                         0.35 + rng.random() * 0.2
            tpsrs = rng.binomial(1, base_tpsr, num_trials).tolist()
            trials = []
            for success in tpsrs:
                trials.append({
                    "goal_achieved": bool(success),
                    "num_llm_calls": int(rng.integers(8, 18)),
                    "num_mp_calls":  int(rng.integers(20, 55)),
                    "total_time":    round(float(rng.uniform(80, 150)), 1),
                    "wasted_real_actions": 0 if (is_fs and "rehearsal" not in disable) else int(rng.integers(0, 5)),
                    "rehearsal_saves": int(rng.integers(1, 4)) if (is_fs and "rehearsal" not in disable) else 0,
                    "cascade_depth":   round(float(rng.uniform(1.2, 2.5)), 1) if (is_fs and "cascade" not in disable) else 0.0,
                    "acfm_accuracy":   round(float(rng.uniform(0.75, 0.92)), 2) if (is_fs and "acfm" not in disable) else 0.0,
                })
            results[m][sid_short] = summarise_trials(trials)
    return results

# ─────────────────────────── CONSOLE TABLES ───────────────────

def fmt(v, decimals=1, na="--"):
    if v is None:
        return na
    return f"{v:.{decimals}f}"


def print_selected_summary(
    results: dict,
    methods: list[str],
    groups: list[str],
    scenario_ids: list[str] | None = None,
):
    """Print a compact table for the exact methods/scenarios requested."""
    sids = scenario_ids or all_group_sids(groups)
    if not sids:
        return
    width = 31 + 18 * len(sids) + 29
    print("\n" + "=" * width)
    print("SELECTED RUN SUMMARY")
    header = f"{'Method':<24}"
    for sid in sids:
        header += f" {sid + ' TPSR':>8} {sid + ' W':>7}"
    header += f" {'Avg TPSR':>9} {'Avg W':>7}"
    print(header)
    print("-" * width)

    for method in methods:
        mdata = results.get(method, {})
        row = f"{method:<24}"
        tpsrs, wasted = [], []
        for sid in sids:
            s = mdata.get(sid, {})
            t = s.get("TPSR")
            w = s.get("avg_wasted")
            row += f" {fmt(t):>8} {fmt(w):>7}"
            if t is not None:
                tpsrs.append(t)
            if w is not None:
                wasted.append(w)
        avg_t = float(np.mean(tpsrs)) if tpsrs else None
        avg_w = float(np.mean(wasted)) if wasted else None
        row += f" {fmt(avg_t):>9} {fmt(avg_w):>7}"
        print(row)
    print("=" * width)


# ─────────────────────────── CSV EXPORT ───────────────────────

def save_csv(all_results: dict, path: Path):
    import csv
    rows = []
    for m in METHODS:
        for sid, s in all_results.get(m, {}).items():
            rows.append({
                "method": m, "scenario": sid,
                **{k: v for k, v in s.items() if isinstance(v, (int, float, str, bool))},
            })
    if not rows:
        return
    preferred = [
        "method", "scenario", "error",
        "TPSR", "TPSR_std", "avg_llm", "avg_mp", "avg_time", "avg_wasted",
        "avg_rehearsal", "avg_cascade", "avg_acfm_acc", "n",
    ]
    all_keys = []
    for key in preferred:
        if any(key in row for row in rows):
            all_keys.append(key)
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved CSV → {path}")


# ─────────────────────────── MAIN ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run planning methods on the ForeSight-TAMP scenarios."
    )
    parser.add_argument("--groups", nargs="*", default=list(GROUP_SCENARIOS),
                        choices=list(GROUP_SCENARIOS),
                        help="Scenario groups to evaluate (default: all five)")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=(
            "Exact scenario stems or unambiguous prefixes to evaluate. "
            "When supplied, this overrides --groups."
        ),
    )
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Method names to run (default: the five compared methods). E.g. --methods ForeSight LLM3-FS")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials per scenario per method (the paper uses 3 per model)")
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=10,
        help=(
            "Hard maximum number of actual chat-completion API requests per "
            "trial, including explicit transport retries."
        ),
    )
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="LLM model (default: gpt-3.5-turbo). Examples: gpt-4o, qwen3:8b")
    parser.add_argument(
        "--model-key",
        default=None,
        help=(
            "Registry key from configs/model_registry.py. Resolves the model "
            "identifier, endpoint, context length, timeout, response-format "
            "policy and token pricing together, so a multi-model study is one "
            "flag per run instead of six. Overrides the individual flags."
        ),
    )
    parser.add_argument("--llm-base-url", default=None,
                        help="OpenAI-compatible API base URL. Example: http://localhost:11434/v1 for Ollama")
    parser.add_argument("--llm-api-key", default=None,
                        help="API key for the selected endpoint. Use 'ollama' for local Ollama.")
    parser.add_argument("--llm-response-format", choices=["auto", "on", "off"], default="auto",
                        help="Whether to send OpenAI response_format. Auto disables it for local Ollama endpoints.")
    parser.add_argument(
        "--expected-resolved-model",
        default=None,
        help=(
            "Abort if an API response reports a different resolved model ID. "
            "Use this to freeze a dated model snapshot."
        ),
    )
    parser.add_argument(
        "--llm-input-cost-per-million",
        type=float,
        default=None,
        help="Pricing snapshot used only for recorded cost calculation.",
    )
    parser.add_argument(
        "--llm-output-cost-per-million",
        type=float,
        default=None,
        help="Pricing snapshot used only for recorded cost calculation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260825,
        help="Base seed; each scenario/trial pair shares a seed across methods.",
    )
    parser.add_argument(
        "--no-capture-llm-payloads",
        action="store_true",
        help="Store only prompt/response hashes instead of exact payloads.",
    )
    parser.add_argument(
        "--no-acfm-maps",
        action="store_true",
        help="Skip rendering per-step ACFM heatmaps.",
    )
    parser.add_argument(
        "--no-trajectory-artifacts",
        action="store_true",
        help="Skip writing executed joint/Cartesian path files.",
    )
    parser.add_argument(
        "--require-clean-worktree",
        action="store_true",
        help="Refuse to start unless the Git worktree is clean.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (no LLM calls), for testing the pipeline")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Re-use a previous run's directory to resume it.")
    parser.add_argument("--resume", default=None, metavar="DIR",
                        help="Alias for --out-dir: resume an interrupted run from this directory.")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore existing results.json/raw_trials.json in --out-dir and run selected trials again.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show each method sequentially in the PyBullet GUI.",
    )
    parser.add_argument(
        "--play-traj",
        action="store_true",
        help="Animate planned trajectories; requires --gui.",
    )
    args = parser.parse_args()

    if args.model_key:
        from configs.model_registry import get_model
        try:
            _entry = get_model(args.model_key)
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        args.model = _entry["model"]
        args.llm_base_url = _entry.get("base_url") or args.llm_base_url
        args.llm_api_key = _entry.get("api_key") or args.llm_api_key
        args.llm_response_format = _entry.get(
            "response_format", args.llm_response_format)
        if _entry.get("expected_resolved_model"):
            args.expected_resolved_model = _entry["expected_resolved_model"]
        if args.llm_input_cost_per_million is None:
            args.llm_input_cost_per_million = _entry["input_cost_per_million"]
        if args.llm_output_cost_per_million is None:
            args.llm_output_cost_per_million = _entry["output_cost_per_million"]
        os.environ["FORESIGHT_LLM_TIMEOUT"] = str(_entry["timeout_seconds"])
        os.environ["FORESIGHT_LLM_MAX_TOKENS"] = str(_entry["max_tokens"])
        os.environ["FORESIGHT_LLM_CONTEXT_TOKENS"] = str(
            _entry["context_tokens"])

    if args.play_traj and not args.gui:
        parser.error("--play-traj requires --gui")
    if args.max_llm_calls <= 0:
        parser.error("--max-llm-calls must be positive")
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if (args.llm_input_cost_per_million is None) != (
        args.llm_output_cost_per_million is None
    ):
        parser.error(
            "Specify both --llm-input-cost-per-million and "
            "--llm-output-cost-per-million, or neither."
        )

    # --resume is a convenient alias for --out-dir, but two different targets
    # are almost certainly an operator error.
    if args.resume and args.out_dir and Path(args.resume) != Path(args.out_dir):
        parser.error("--resume and --out-dir refer to different directories")
    if args.resume and not args.out_dir:
        args.out_dir = args.resume

    response_format = None
    if args.llm_response_format == "on":
        response_format = "1"
    elif args.llm_response_format == "off":
        response_format = "0"
    os.environ["FORESIGHT_LLM_MODEL"] = args.model
    if args.llm_base_url:
        os.environ["FORESIGHT_LLM_BASE_URL"] = args.llm_base_url
    if args.llm_api_key:
        os.environ["FORESIGHT_LLM_API_KEY"] = args.llm_api_key
    if response_format is not None:
        os.environ["FORESIGHT_LLM_RESPONSE_FORMAT"] = response_format

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_BASE / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = _install_run_log(out_dir)
    logger.info(f"Output directory: {out_dir}")

    methods_to_run = args.methods or DEFAULT_METHODS
    # Validate
    for m in methods_to_run:
        if m not in METHODS:
            parser.error(f"Unknown method '{m}'. Choose from: {list(METHODS.keys())}")

    try:
        scenarios_to_run = (
            load_selected_scenarios(args.scenarios)
            if args.scenarios
            else [
                scenario
                for group in args.groups
                for scenario in load_group(group)
            ]
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not scenarios_to_run:
        parser.error("No scenarios found. Check --scenarios or --groups.")
    selected_sids = [sid for sid, _ in scenarios_to_run]

    code_provenance = _code_provenance()
    global RENDER_ACFM_MAPS, EXPORT_TRAJECTORY_ARTIFACTS
    RENDER_ACFM_MAPS = not args.no_acfm_maps
    EXPORT_TRAJECTORY_ARTIFACTS = not args.no_trajectory_artifacts

    if args.require_clean_worktree and code_provenance["git_dirty"]:
        parser.error(
            "The worktree is dirty. Freeze/commit the benchmark code and inputs "
            "before starting this run."
        )

    requested_config = {
        "protocol_schema": "foresight-benchmark-v2",
        "model": args.model,
        "expected_resolved_model": args.expected_resolved_model,
        "llm_base_url": args.llm_base_url,
        "llm_response_format": args.llm_response_format,
        "llm_input_cost_per_million": args.llm_input_cost_per_million,
        "llm_output_cost_per_million": args.llm_output_cost_per_million,
        "trials_per_method_scenario": args.trials,
        # Alias of ``max_llm_api_requests_per_trial``.
        "max_llm_calls_per_trial": args.max_llm_calls,
        "max_llm_api_requests_per_trial": args.max_llm_calls,
        "call_budget_definition": (
            "actual chat-completion requests initiated by this process, "
            "including explicit transport retries; SDK retries disabled"
        ),
        "methods": methods_to_run,
        "method_configurations": {
            name: protocol_method_config(name) for name in methods_to_run
        },
        "matched_rollout_budget_definition": (
            "Abl_matched_rollout pays ACFM construction cost, samples a "
            "deterministic score/mask-independent raw parameter domain in "
            "non-overlapping low-discrepancy grid batches across planning "
            "attempts (the exact push goal and finite discrete domains may "
            "repeat), and "
            "uses at most place=10, push=9, stacking-place=1, and "
            "Hanoi-place=2 simulator candidate rollouts per parameterized "
            "action. It chooses the lowest stable passing candidate ID and "
            "makes zero selector LLM calls; total compute is not claimed equal."
        ),
        "scenarios": selected_sids,
        "scenario_groups": None if args.scenarios else args.groups,
        "base_seed": args.seed,
        "seed_schedule": (
            "base_seed + scenario_index * trials_per_method_scenario + trial_index; "
            "paired across methods"
        ),
        "dry_run": args.dry_run,
        "use_gui": args.gui,
        "play_traj": args.play_traj,
        "capture_llm_payloads": not args.no_capture_llm_payloads,
        "run_log": str(run_log_path),
        "discarded_simulator_primitives_definition": (
            "successful committed simulator primitives from unsuccessful "
            "attempts whose state is discarded by reset or episode failure; "
            "failed motion-planning attempts and rehearsal rollout primitives "
            "are excluded"
        ),
        "code_provenance": code_provenance,
    }
    config_path = out_dir / "run_config.json"
    existing_config = None
    if config_path.exists() and not args.force_rerun:
        with config_path.open(encoding="utf-8") as stream:
            existing_config = json.load(stream)
        mismatches = {
            key: {"existing": existing_config.get(key), "requested": value}
            for key, value in requested_config.items()
            if existing_config.get(key) != value
        }
        if mismatches:
            formatted = json.dumps(mismatches, indent=2, default=str)
            parser.error(
                "Refusing to resume with a different protocol. Use a fresh "
                f"output directory. Mismatches:\n{formatted}"
            )
        run_config = existing_config
        run_config.setdefault("resume_history", []).append(
            datetime.now().isoformat(timespec="seconds")
        )
    else:
        protocol_snapshot = _archive_protocol_inputs(out_dir, selected_sids)
        run_config = {
            **requested_config,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "protocol_snapshot": protocol_snapshot,
            "resume_history": [],
        }
    _atomic_write_json(config_path, run_config)

    if args.dry_run:
        logger.info("DRY RUN — using mock results (no LLM calls)")
        all_results = make_mock_results(
            args.groups,
            args.trials,
            scenario_ids=selected_sids if args.scenarios else None,
            methods=methods_to_run,
        )
    else:
        from utils.metrics_util import trial_trace

        # Patch LLM model before importing any in-repo planner/baseline.
        import utils.llm_util as llm_util
        from utils.llm_util import GPT_Chat
        llm_util.configure_llm_runtime(
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            response_format=response_format,
        )
        def patched_init(self, use_gpt_4=True, *a, **kw):
            self.llm_gpt = GPT_Chat(engine=args.model)
        llm_util.LLMBase.__init__ = patched_init
        logger.info(
            "LLM model → %s%s",
            args.model,
            f" ({args.llm_base_url})" if args.llm_base_url else "",
        )

        # ── Resume: load any previously saved state ────────────
        checkpoint_path = out_dir / "results.json"
        raw_trials_path = out_dir / "raw_trials.json"

        all_results: dict = {}
        raw_trials: dict  = {}   # {method: {sid: [trial_dict, ...]}}

        if args.force_rerun:
            logger.info("FORCE RERUN — ignoring existing checkpoint files in output directory.")
        elif checkpoint_path.exists() != raw_trials_path.exists():
            raise RuntimeError(
                "Refusing an incomplete resume: results.json and "
                "raw_trials.json must either both exist or both be absent"
            )
        elif checkpoint_path.exists():
            with open(checkpoint_path) as f:
                all_results = json.load(f)
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        if not args.force_rerun and raw_trials_path.exists():
            with open(raw_trials_path) as f:
                raw_trials = json.load(f)
            logger.info(f"Loaded {sum(len(v) for v in raw_trials.values())} "
                        f"raw trial sets from previous run.")
            _validate_resume_checkpoint(
                raw_trials,
                out_dir=out_dir,
                methods=methods_to_run,
                scenarios=selected_sids,
                trials_per_cell=args.trials,
                max_llm_requests=args.max_llm_calls,
                base_seed=args.seed,
            )
            # Treat raw, hash-validated trial records as authoritative.  This
            # prevents a stale/corrupted summary file from surviving another
            # interruption before the final full rebuild.
            all_results = {}
            for method_name, scenario_rows in raw_trials.items():
                all_results[method_name] = {
                    scenario: summarise_trials(rows)
                    for scenario, rows in scenario_rows.items()
                    if rows
                }

        def _save_checkpoint():
            """Atomically write both checkpoint files."""
            tmp = checkpoint_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            tmp.replace(checkpoint_path)

            tmp2 = raw_trials_path.with_suffix(".json.tmp")
            with open(tmp2, "w") as f:
                json.dump(raw_trials, f, indent=2, default=str)
            tmp2.replace(raw_trials_path)

        total = len(methods_to_run) * len(scenarios_to_run) * args.trials
        done  = 0

        for m in methods_to_run:
            cfg = METHODS[m]
            all_results.setdefault(m, {})
            raw_trials.setdefault(m, {})

            for scenario_index, (sid_short, scenario) in enumerate(
                scenarios_to_run
            ):
                # How many trials were already completed for this cell?
                completed_trials = raw_trials[m].get(sid_short, [])
                n_done = len(completed_trials)
                n_remaining = args.trials - n_done

                done += n_done  # count already-finished trials toward progress

                if n_remaining <= 0:
                    logger.info(f"  SKIP  {m} × {sid_short} "
                                f"(already have {n_done}/{args.trials} trials)")
                    continue

                if n_done > 0:
                    logger.info(f"  RESUME {m} × {sid_short} "
                                f"({n_done}/{args.trials} done, running {n_remaining} more)")

                for trial in range(n_done, args.trials):
                    done += 1
                    logger.info(f"[{done}/{total}] {m} × {sid_short} "
                                f"trial {trial+1}/{args.trials}")
                    trial_seed = (
                        args.seed + scenario_index * args.trials + trial
                    )
                    random.seed(trial_seed)
                    np.random.seed(trial_seed % (2**32))
                    artifact_relative = None
                    artifact_directory = None
                    if cfg["class"] == "foresight":
                        artifact_relative = (
                            Path("artifacts") / m / sid_short
                            / f"trial_{trial:03d}"
                        )
                        artifact_directory = out_dir / artifact_relative
                        if artifact_directory.exists():
                            raise RuntimeError(
                                "Refusing to mix a new trial with a pre-existing "
                                f"artifact directory: {artifact_directory}"
                            )
                        artifact_directory.mkdir(parents=True)
                    with trial_trace(
                        max_llm_requests=args.max_llm_calls,
                        input_cost_per_million=args.llm_input_cost_per_million,
                        output_cost_per_million=args.llm_output_cost_per_million,
                        capture_llm_payloads=not args.no_capture_llm_payloads,
                    ) as metrics:
                        r = run_trial(
                            m,
                            cfg,
                            scenario,
                            args.max_llm_calls,
                            use_gui=args.gui,
                            play_traj=args.play_traj,
                            artifact_dir=artifact_directory,
                        )

                    metric_summary = metrics.summary()
                    reported_planner_calls = r.get("num_llm_calls", 0)
                    reported_motion_planner_calls = r.get("num_mp_calls", 0)
                    r.update(metric_summary)
                    r["reported_planner_calls"] = reported_planner_calls
                    r["reported_motion_planner_calls"] = (
                        reported_motion_planner_calls
                    )
                    # Actual API requests, including retries.
                    r["num_llm_calls"] = metric_summary["llm_api_requests"]
                    r["num_mp_calls"] = metric_summary["motion_planner_calls"]
                    r["trial_index"] = trial
                    r["trial_seed"] = trial_seed
                    if artifact_relative is not None:
                        r["artifact_directory"] = artifact_relative.as_posix()
                        r["artifact_files_sha256"] = _directory_hashes(
                            artifact_directory
                        )

                    trace_payload = {
                        "method": m,
                        "scenario": sid_short,
                        "trial_index": trial,
                        "trial_seed": trial_seed,
                        "trace": metrics.export(),
                    }
                    trace_path = (
                        out_dir / "traces" / m / sid_short
                        / f"trial_{trial:03d}.json"
                    )
                    _atomic_write_json(trace_path, trace_payload)
                    r["trace_file"] = str(trace_path.relative_to(out_dir))
                    r["trace_sha256"] = _sha256_file(trace_path)

                    resolved_models = set(metric_summary["resolved_models"])
                    model_mismatch = (
                        args.expected_resolved_model is not None
                        and resolved_models
                        and resolved_models != {args.expected_resolved_model}
                    )
                    if model_mismatch:
                        r["protocol_violation"] = {
                            "expected_resolved_model": args.expected_resolved_model,
                            "observed_resolved_models": sorted(resolved_models),
                        }
                        r.setdefault(
                            "error",
                            "Resolved model did not match the frozen protocol",
                        )
                    completed_trials.append(r)
                    raw_trials[m][sid_short] = completed_trials

                    # Summarise from all trials collected so far and checkpoint
                    summary = summarise_trials(completed_trials)
                    summary["_trial_acfm"] = [
                        t.get("acfm_accuracy", 0.0) for t in completed_trials
                    ]
                    all_results[m][sid_short] = summary
                    _save_checkpoint()
                    if model_mismatch:
                        raise RuntimeError(
                            "Resolved model mismatch; checkpointed the violating "
                            "trial and aborted to prevent a mixed-model run."
                        )

        logger.info("All trials complete.")

        # Rebuild summaries from raw trials on every run/resume. This keeps
        # result schemas current when the summarisation code gains new fields.
        for method_name, scenarios in raw_trials.items():
            all_results.setdefault(method_name, {})
            for sid_short, trials in scenarios.items():
                if not trials:
                    continue
                summary = summarise_trials(trials)
                summary["_trial_acfm"] = [
                    t.get("acfm_accuracy", 0.0) for t in trials
                ]
                all_results[method_name][sid_short] = summary
        _save_checkpoint()

    # ── Console tables ─────────────────────────────────────────
    print_selected_summary(
        all_results,
        methods_to_run,
        args.groups,
        scenario_ids=selected_sids if args.scenarios else None,
    )

    # ── CSV ────────────────────────────────────────────────────
    save_csv(all_results, out_dir / "results_summary.csv")

    # ── Final JSON (also written per-trial during real runs) ───
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nDone. Results → {out_dir}")
    logger.info("  results.json             — summaries (also checkpointed per trial)")
    logger.info("  raw_trials.json          — per-trial raw data (used for resume)")
    logger.info("  results_summary.csv      — flat CSV")
    logger.info("  run_config.json          — reproducibility settings")
    logger.info("  run.log                  — complete Python logging stream")
    logger.info("")
    logger.info("To resume an interrupted run:")
    resume_cmd = [
        "python3",
        "generate_results.py",
        "--resume",
        str(out_dir),
        "--trials",
        str(args.trials),
        "--max-llm-calls",
        str(args.max_llm_calls),
        "--model",
        args.model,
        "--seed",
        str(args.seed),
    ]
    resume_cmd += ["--methods", *methods_to_run]
    if args.scenarios:
        resume_cmd += ["--scenarios", *selected_sids]
    else:
        resume_cmd += ["--groups", *args.groups]
    if args.llm_base_url:
        resume_cmd += ["--llm-base-url", args.llm_base_url]
    if args.expected_resolved_model:
        resume_cmd += [
            "--expected-resolved-model", args.expected_resolved_model]
    if args.llm_input_cost_per_million is not None:
        resume_cmd += [
            "--llm-input-cost-per-million",
            str(args.llm_input_cost_per_million),
            "--llm-output-cost-per-million",
            str(args.llm_output_cost_per_million),
        ]
    if args.llm_response_format != "auto":
        resume_cmd += ["--llm-response-format", args.llm_response_format]
    if args.no_capture_llm_payloads:
        resume_cmd += ["--no-capture-llm-payloads"]
    if args.require_clean_worktree:
        resume_cmd += ["--require-clean-worktree"]
    if args.dry_run:
        resume_cmd += ["--dry-run"]
    if args.gui:
        resume_cmd += ["--gui"]
    if args.play_traj:
        resume_cmd += ["--play-traj"]
    logger.info("  " + " ".join(resume_cmd))
    if args.llm_api_key:
        logger.info(
            "  API key omitted from the resume command; provide it through a "
            "protected environment variable."
        )


if __name__ == "__main__":
    main()
