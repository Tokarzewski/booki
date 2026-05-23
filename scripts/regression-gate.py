#!/usr/bin/env python3
"""Fail a CI run when a per-benchmark regression exceeds --max-regression.

Both inputs are the JSON emitted by `booki_bench --json`, expected shape:

    {
      "benchmarks": [
        { "name": "...", "ms_per_token": <float>, "tokens_per_sec": <float> },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text())
    return {b["name"]: b for b in raw["benchmarks"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--current", required=True, type=Path)
    p.add_argument(
        "--max-regression",
        type=float,
        default=0.10,
        help="Fail if any benchmark gets slower by more than this fraction (default: 0.10 = 10%%).",
    )
    args = p.parse_args()

    baseline = load(args.baseline)
    current = load(args.current)

    rows: list[tuple[str, float, float, float]] = []
    failed = []
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
            rows.append((name, float("nan"), cur["ms_per_token"], float("nan")))
            continue
        delta = (cur["ms_per_token"] - base["ms_per_token"]) / base["ms_per_token"]
        rows.append((name, base["ms_per_token"], cur["ms_per_token"], delta))
        if delta > args.max_regression:
            failed.append((name, delta))

    print(f"{'benchmark':<40} {'baseline':>10} {'current':>10} {'delta':>10}")
    for name, b, c, d in rows:
        delta_str = "new" if d != d else f"{d:+.1%}"
        b_str = "-" if b != b else f"{b:.2f}ms"
        print(f"{name:<40} {b_str:>10} {c:.2f}ms {delta_str:>10}")

    if failed:
        print("\n::error::Regressions exceeding threshold:", file=sys.stderr)
        for name, d in failed:
            print(f"  {name}: {d:+.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
