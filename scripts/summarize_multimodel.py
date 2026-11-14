#!/usr/bin/env python3
"""Fold per-model benchmark directories into one scenario-by-model view.

``run_multi_model_benchmark.py`` writes each model into its own directory so
runs never collide. That isolation is what makes the question "which model
solved which scenario" awkward to answer by eye, so this reassembles them.

A cell is read as solved when TPSR is above zero. At one trial per cell that is
simply pass/fail; at ten it is the success rate, and ``--metric`` exposes the
other per-cell numbers the same way.

Usage::

    scripts/summarize_multimodel.py results/smoke_multimodel
    scripts/summarize_multimodel.py results/multimodel --metric avg_wasted
    scripts/summarize_multimodel.py results/smoke_multimodel --csv out.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from configs.model_registry import get_model, model_keys
except Exception:  # noqa: BLE001 - summarising must not need the registry
    get_model = None
    model_keys = None


def label_for(key):
    if get_model is None:
        return key
    try:
        return get_model(key)["label"]
    except Exception:  # noqa: BLE001
        return key


def ordered_keys(found):
    """Registry order where known, then anything else alphabetically."""
    if model_keys is None:
        return sorted(found)
    known = [key for key in model_keys(include_smoke=True) if key in found]
    return known + sorted(found - set(known))


def load(run_dir):
    """model_key -> {method -> {scenario -> metrics}}, skipping empty dirs."""
    loaded = {}
    for path in sorted(run_dir.glob("*/results.json")):
        try:
            loaded[path.parent.name] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  skipping {path}: {exc}", file=sys.stderr)
    return loaded


def render(results, method, metric, scenarios, keys):
    width = max([len(s) for s in scenarios] + [8])
    header = f"{'scenario':<{width}}" + "".join(
        f"  {key[:14]:>14}" for key in keys)
    lines = [header, "-" * len(header)]

    solved = {key: 0 for key in keys}
    counted = {key: 0 for key in keys}
    for scenario in scenarios:
        row = f"{scenario:<{width}}"
        for key in keys:
            cell = results.get(key, {}).get(method, {}).get(scenario)
            if cell is None:
                row += f"  {'-':>14}"
                continue
            value = cell.get(metric)
            if value is None:
                row += f"  {'n/a':>14}"
                continue
            counted[key] += 1
            if metric == "TPSR":
                ok = value > 0
                solved[key] += 1 if ok else 0
                mark = "PASS" if ok else "fail"
                row += f"  {mark + f' {value:5.1f}%':>14}"
            else:
                row += f"  {value:>14.2f}"
        lines.append(row)

    if metric == "TPSR":
        lines.append("-" * len(header))
        total = f"{'solved':<{width}}"
        for key in keys:
            total += f"  {f'{solved[key]}/{counted[key]}':>14}"
        lines.append(total)
    return "\n".join(lines), solved, counted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--method", default="ForeSight")
    parser.add_argument("--metric", default="TPSR")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"not a directory: {args.run_dir}")

    results = load(args.run_dir)
    if not results:
        print(f"No results.json found under {args.run_dir}")
        return 1

    keys = ordered_keys(set(results))
    methods = sorted({m for r in results.values() for m in r})
    if args.method not in methods:
        parser.error(f"method {args.method!r} not in results; have: "
                     f"{', '.join(methods) or '(none)'}")

    scenarios = []
    for key in keys:
        for scenario in results[key].get(args.method, {}):
            if scenario not in scenarios:
                scenarios.append(scenario)

    print(f"{args.method} — {args.metric}   ({args.run_dir})\n")
    for key in keys:
        print(f"  {key:<16} {label_for(key)}")
    print()
    table, solved, counted = render(
        results, args.method, args.metric, scenarios, keys)
    print(table)

    if args.metric == "TPSR":
        print()
        for key in keys:
            done = counted[key]
            missing = len(scenarios) - done
            note = f"  ({missing} scenario(s) missing)" if missing else ""
            rate = 100.0 * solved[key] / done if done else 0.0
            print(f"  {label_for(key):<26} {solved[key]}/{done} "
                  f"scenarios solved  ({rate:.0f}%){note}")

    if args.csv:
        rows = ["model_key,model_label,scenario,metric,value"]
        for key in keys:
            for scenario in scenarios:
                cell = results.get(key, {}).get(args.method, {}).get(scenario)
                value = "" if cell is None else cell.get(args.metric, "")
                rows.append(f'{key},"{label_for(key)}",{scenario},'
                            f'{args.metric},{value}')
        args.csv.write_text("\n".join(rows) + "\n")
        print(f"\ncsv -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
