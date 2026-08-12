"""Timing and percentile reporting.

The single rule of this module: never report a mean.

A mean callback time of 3 ms with a p99 of 180 ms means one beat in a
hundred lands visibly late. On a drop, that is the beat the room
remembers. Tail latency is the whole story in realtime systems, so
every number that leaves this project is a distribution.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# Percentiles reported everywhere in this project.
PCTS = (50, 90, 95, 99, 99.9)


def summarize(values_ns: np.ndarray, scale: str = "us") -> dict:
    """Turn an array of nanosecond durations into a percentile summary.

    Args:
        values_ns: 1-D array of durations in nanoseconds.
        scale: "ns", "us" or "ms" — the unit of the returned numbers.

    Returns:
        dict with n, unit, min, max, mean (for reference only) and p50..p99.9.
    """
    divisor = {"ns": 1.0, "us": 1e3, "ms": 1e6}[scale]
    v = np.asarray(values_ns, dtype=np.float64) / divisor

    if v.size == 0:
        return {"n": 0, "unit": scale}

    out = {"n": int(v.size), "unit": scale,
           "min": float(v.min()), "max": float(v.max()),
           "mean": float(v.mean())}
    for p in PCTS:
        out[f"p{p:g}"] = float(np.percentile(v, p))
    return out


def print_summary(label: str, summary: dict) -> None:
    """Print one summary line, aligned so a sweep reads as a table."""
    if summary.get("n", 0) == 0:
        print(f"{label:<18s}  (no samples)")
        return
    u = summary["unit"]
    print(
        f"{label:<18s} n={summary['n']:>7d}  "
        f"p50={summary['p50']:>9.1f}  "
        f"p95={summary['p95']:>9.1f}  "
        f"p99={summary['p99']:>9.1f}  "
        f"max={summary['max']:>9.1f}  {u}"
    )


def buffer_latency_ms(blocksize: int, samplerate: int) -> float:
    """The latency floor imposed by the audio buffer, before any of our code.

    The first sample in a block was captured `blocksize / samplerate`
    seconds before the callback fires. Nothing we write can recover it.
    """
    return blocksize / samplerate * 1000.0


def save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_csv(path: str | Path, rows: list[dict]) -> Path:
    """Write a list of flat dicts as CSV. Used for sweep tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render rows as a markdown table, for pasting straight into the README."""
    if not rows:
        return "(no rows)"
    cols = columns or list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])
