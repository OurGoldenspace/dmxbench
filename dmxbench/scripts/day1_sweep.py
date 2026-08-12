#!/usr/bin/env python3
"""Day 1, second half: sweep the blocksize to find the latency/stability curve.

    python scripts/day1_sweep.py
    python scripts/day1_sweep.py --hostapi WASAPI --exclusive --latency low
    python scripts/day1_sweep.py --duration 20 --blocksizes 64 128 256 512

This produces the first real result of the project: the smallest buffer
your machine can sustain without dropouts, and what that buffer costs you
in milliseconds. Paste the markdown table straight into the README.

The verdict is judged on MAX callback duration, not p99. A p99 that fits
comfortably still allows one overrun in a hundred, and one overrun is an
audible click.
"""

import argparse
from datetime import datetime
from pathlib import Path

from dmxbench import (capture, find_input_device, markdown_table, save_csv,
                      summarize, wasapi_exclusive_settings)

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = ["blocksize", "block_period_ms", "driver_latency_ms", "callbacks",
           "overflows", "p50_us", "p95_us", "p99_us", "max_us",
           "p99_pct_period", "max_pct_period", "headroom_ms", "verdict"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=20.0,
                   help="seconds per blocksize")
    p.add_argument("--samplerate", type=int, default=48_000)
    p.add_argument("--blocksizes", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--device", default=None)
    p.add_argument("--hostapi", default=None,
                   help="e.g. WASAPI or ASIO on Windows")
    p.add_argument("--exclusive", action="store_true",
                   help="WASAPI exclusive mode (Windows)")
    p.add_argument("--latency", default="low",
                   help="'low', 'high', or seconds")
    p.add_argument("--tag", default="", help="label for the saved CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = int(args.device) if (args.device or "").isdigit() else args.device

    if args.hostapi:
        found = find_input_device(args.hostapi)
        if found is None:
            raise SystemExit(f"no input device for host API {args.hostapi!r}")
        device = found
        print(f"host API {args.hostapi!r} -> device index {device}")

    extra = wasapi_exclusive_settings() if args.exclusive else None
    latency = float(args.latency) if _is_number(args.latency) else args.latency
    print(f"latency={latency!r}  exclusive={bool(extra)}\n")

    rows = []
    for bs in args.blocksizes:
        print(f"--- blocksize {bs} ({args.duration:.0f}s) ...", flush=True)
        try:
            res = capture(duration_s=args.duration, samplerate=args.samplerate,
                          blocksize=bs, device=device, extra_settings=extra,
                          latency=latency)
        except Exception as exc:  # unsupported blocksize, device busy, etc.
            print(f"    FAILED: {exc}")
            rows.append({c: "" for c in COLUMNS}
                        | {"blocksize": bs, "verdict": f"failed: {exc}"})
            continue

        s = summarize(res["durations_ns"], scale="us")
        period_us = res["block_period_ms"] * 1000.0
        p99_pct = s["p99"] / period_us * 100
        max_pct = s["max"] / period_us * 100
        headroom = res["block_period_ms"] - s["max"] / 1000.0

        # Judge on worst case. p99 "clean" with a max at 70% of the period
        # is a config that will glitch the moment real work goes in the callback.
        if res["input_overflows"] > 0:
            verdict = "DROPOUTS"
        elif max_pct > 50:
            verdict = "fragile"
        elif max_pct > 25:
            verdict = "tight"
        else:
            verdict = "clean"

        rows.append({
            "blocksize": bs,
            "block_period_ms": round(res["block_period_ms"], 3),
            "driver_latency_ms": round(res["driver_latency_ms"], 3),
            "callbacks": res["callbacks"],
            "overflows": res["input_overflows"],
            "p50_us": round(s["p50"], 1),
            "p95_us": round(s["p95"], 1),
            "p99_us": round(s["p99"], 1),
            "max_us": round(s["max"], 1),
            "p99_pct_period": round(p99_pct, 1),
            "max_pct_period": round(max_pct, 1),
            "headroom_ms": round(headroom, 3),
            "verdict": verdict,
        })
        print(f"    period={res['block_period_ms']:.2f}ms  "
              f"driver={res['driver_latency_ms']:.2f}ms  "
              f"p99={s['p99']:.0f}us  max={s['max']:.0f}us "
              f"({max_pct:.0f}% of period)  -> {verdict}")

    print("\n" + markdown_table(rows, COLUMNS))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    excl = "-excl" if args.exclusive else "-shared"
    tag = f"-{args.tag}" if args.tag else ""
    path = save_csv(RESULTS / f"day1-sweep{excl}-{args.latency}{tag}-{stamp}.csv", rows)
    print(f"\nsaved -> {path}")
    print("\nPick the smallest blocksize with verdict=clean. Anything marked\n"
          "fragile has no room left once real feature extraction goes in.")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()