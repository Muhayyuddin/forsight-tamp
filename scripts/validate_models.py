#!/usr/bin/env python3
"""Check that every registry model answers before a benchmark is launched.

Each model gets one trivial completion and one realistic planning-shaped
prompt, and the JSON compliance of the latter is reported, since open-weight
models are markedly worse at emitting strict JSON.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.model_registry import get_model, model_keys  # noqa: E402

PLANNING_PROBE = (
    "You are a robot manipulation planner. Reply with ONLY a JSON list, no "
    "prose and no code fences.\n"
    "Place one object into a basket.\n"
    "Output format: [{\"action\": \"pick\", \"object\": \"box\"}, "
    "{\"action\": \"place\", \"object\": \"box\", \"x\": 0.6, \"y\": 0.0}]\n"
)


def _extract_json(text):
    """Accept a plan wrapped in fences or prose, as local models often emit."""
    if not text:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith(("[", "{")):
                cleaned = candidate
                break
    start = min(
        (i for i in (cleaned.find("["), cleaned.find("{")) if i != -1),
        default=-1)
    if start == -1:
        return None
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def check(key, probes):
    entry = get_model(key)
    import utils.llm_util as llm_util
    os.environ["FORESIGHT_LLM_TIMEOUT"] = str(entry["timeout_seconds"])
    os.environ["FORESIGHT_LLM_MAX_TOKENS"] = str(entry["max_tokens"])
    result = {
        "model_key": key,
        "label": entry["label"],
        "kind": entry["kind"],
        "model": entry["model"],
        "reachable": False,
        "json_ok": 0,
        "json_attempts": probes,
        "latency_s": None,
        "error": None,
    }
    try:
        chat = llm_util.GPT_Chat(
            engine=entry["model"],
            base_url=entry.get("base_url"),
            api_key=entry.get("api_key"),
        )
        started = time.time()
        ok, out = chat.get_response(prompt=None, messages=[
            {"role": "user", "content": "Reply with the single word: ready"}])
        result["latency_s"] = round(time.time() - started, 2)
        if not ok:
            result["error"] = "no response from endpoint"
            return result
        result["reachable"] = True
        result["sample"] = (out or "").strip()[:60]

        for _ in range(probes):
            ok, out = chat.get_response(prompt=None, messages=[
                {"role": "user", "content": PLANNING_PROBE}])
            if ok and _extract_json(out) is not None:
                result["json_ok"] += 1
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Registry keys to check (default: all).")
    parser.add_argument("--probes", type=int, default=3,
                        help="Planning-shaped prompts per model.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    keys = args.models or model_keys()
    results = []
    for key in keys:
        print(f"--- {key}", flush=True)
        outcome = check(key, args.probes)
        results.append(outcome)
        status = "OK " if outcome["reachable"] else "FAIL"
        detail = (
            f"json {outcome['json_ok']}/{outcome['json_attempts']}"
            f"  latency {outcome['latency_s']}s"
            if outcome["reachable"] else outcome["error"]
        )
        print(f"    {status} {outcome['label']}: {detail}", flush=True)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")

    failed = [r["model_key"] for r in results if not r["reachable"]]
    if failed:
        print(f"\nUNREACHABLE: {', '.join(failed)}")
        return 1
    print("\nAll checked models responded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
