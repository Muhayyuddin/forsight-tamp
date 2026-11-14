#!/usr/bin/env python3
"""Download, verify and register the local GGUF planner models.

Three properties matter more than speed here:

* **Resumable.** These are 21-43 GB files. A dropped connection must not cost
  the whole transfer, so every download resumes in place.
* **Verified.** The expected byte size comes from the HuggingFace API and is
  checked after transfer; a truncated GGUF otherwise fails later, inside the
  benchmark, where it is far more expensive to diagnose.
* **Recorded.** A local model has no dated snapshot to pin a run against, so
  its SHA-256 is its identity. The hash is written into a provenance file that
  the run manifest can cite.

Usage::

    scripts/fetch_local_models.py --list
    scripts/fetch_local_models.py --models glm47-flash
    scripts/fetch_local_models.py            # all three, smallest first
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DEST = Path.home() / "models" / "gguf"

# Publisher chosen for provenance consistency: the requested file names and
# sizes match unsloth's quantisations exactly, including the "UD-" dynamic
# quant for Qwen. The Llama Q4_K_M is byte-identical in size across every
# publisher that hosts it.
SOURCES = {
    "glm47-flash": {
        "repo": "unsloth/GLM-4.7-Flash-GGUF",
        "file": "GLM-4.7-Flash-Q5_K_M.gguf",
        "expected_gb": 21.41,
    },
    "qwen36-35b-a3b": {
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "file": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
        "expected_gb": 26.46,
    },
    "mistral-small-24b": {
        "repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "file": "Mistral-Small-3.2-24B-Instruct-2506-Q5_K_M.gguf",
        "expected_gb": 16.76,
    },
    "llama33-70b": {
        "repo": "unsloth/Llama-3.3-70B-Instruct-GGUF",
        "file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "expected_gb": 42.52,
    },
}

# Smallest first: a failure surfaces after 21 GB rather than 42 GB, and the
# smallest model is enough to prove the import and serve path.
ORDER = ["mistral-small-24b", "glm47-flash", "qwen36-35b-a3b", "llama33-70b"]


def _api(url):
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def remote_size(repo, filename):
    """Authoritative byte size from the HuggingFace file listing."""
    listing = _api(
        f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1")
    for entry in listing:
        if entry.get("path", "").split("/")[-1] == filename:
            return (entry.get("size")
                    or (entry.get("lfs") or {}).get("size") or 0)
    raise FileNotFoundError(f"{filename} not found in {repo}")


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(key, dest_dir, skip_hash=False):
    source = SOURCES[key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / source["file"]
    url = (f"https://huggingface.co/{source['repo']}/resolve/main/"
           f"{source['file']}?download=true")

    try:
        expected = remote_size(source["repo"], source["file"])
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        print(f"  could not read remote size: {exc}")
        expected = 0

    have = target.stat().st_size if target.exists() else 0
    if expected and have == expected:
        print(f"  already complete: {target.name} "
              f"({have / 1e9:.2f} GB)")
    else:
        if have:
            print(f"  resuming at {have / 1e9:.2f} GB "
                  f"of {expected / 1e9:.2f} GB")
        started = time.time()
        # -C - resumes; --retry survives transient drops mid-transfer.
        result = subprocess.run(
            ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
             "--retry-all-errors", "-o", str(target), url],
            check=False,
        )
        if result.returncode != 0:
            return {"model_key": key, "ok": False,
                    "error": f"curl exit {result.returncode}",
                    "path": str(target)}
        elapsed = max(time.time() - started, 1e-6)
        moved = target.stat().st_size - have
        print(f"  transferred {moved / 1e9:.2f} GB in {elapsed / 60:.1f} min "
              f"({moved / elapsed / 1e6:.1f} MB/s)")

    size = target.stat().st_size
    if expected and size != expected:
        return {"model_key": key, "ok": False,
                "error": f"size mismatch: {size} != {expected}",
                "path": str(target)}

    record = {
        "model_key": key,
        "ok": True,
        "path": str(target),
        "file": source["file"],
        "repo": source["repo"],
        "size_bytes": size,
        "size_gb": round(size / 1e9, 2),
    }
    if not skip_hash:
        print("  hashing (a 40 GB file takes a few minutes)...", flush=True)
        started = time.time()
        record["sha256"] = sha256(target)
        print(f"  sha256 {record['sha256'][:16]}... "
              f"in {(time.time() - started) / 60:.1f} min")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"Subset of: {', '.join(ORDER)}")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--skip-hash", action="store_true",
                        help="Skip SHA-256 (faster, but no provenance record)")
    parser.add_argument("--list", action="store_true",
                        help="Show sources and exit without downloading")
    args = parser.parse_args()

    keys = args.models or ORDER
    unknown = [key for key in keys if key not in SOURCES]
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(unknown)}")

    if args.list:
        total = sum(SOURCES[key]["expected_gb"] for key in keys)
        for key in keys:
            source = SOURCES[key]
            print(f"{key:16s} {source['expected_gb']:>6.2f} GB  "
                  f"{source['repo']}/{source['file']}")
        print(f"{'total':16s} {total:>6.2f} GB  -> {args.dest}")
        return 0

    free_gb = os.statvfs(args.dest.parent if args.dest.exists()
                         else Path.home()).f_bavail * 4096 / 1e9
    needed = sum(SOURCES[key]["expected_gb"] for key in keys)
    print(f"need ~{needed:.1f} GB, free {free_gb:.0f} GB -> {args.dest}\n")
    if free_gb < needed * 1.05:
        print("Refusing: not enough free disk space.")
        return 1

    results = []
    for key in keys:
        print(f"=== {key}  ({SOURCES[key]['expected_gb']} GB)", flush=True)
        record = download(key, args.dest, skip_hash=args.skip_hash)
        results.append(record)
        print("  OK" if record["ok"] else f"  FAILED: {record['error']}",
              flush=True)

    provenance_path = args.dest / "provenance.json"
    existing = {}
    if provenance_path.exists():
        try:
            existing = {
                item["model_key"]: item
                for item in json.loads(provenance_path.read_text())
            }
        except Exception:  # noqa: BLE001
            existing = {}
    for record in results:
        existing[record["model_key"]] = record
    provenance_path.write_text(
        json.dumps(list(existing.values()), indent=2, sort_keys=True))
    print(f"\nprovenance -> {provenance_path}")

    failed = [r["model_key"] for r in results if not r["ok"]]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All requested models present and size-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
