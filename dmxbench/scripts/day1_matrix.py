#!/usr/bin/env python3
"""The config matrix: how much latency is free on this machine?

    python scripts/day1_matrix.py --hostapi WASAPI

Runs {shared, exclusive} x {high, low} at a fixed blocksize and reports
the input path for each. On Windows this typically shows the input path
falling by more than half through configuration alone, before a line of
detection code is written.

sounddevice's own default is latency='high' and shared mode, so the
top-left cell is what you get if you never think about it.
"""

import argparse
from datetime import datetime
from pathlib import Path

from dmxbench import (capture, find_input_device, markdown_table, save_csv,
                      summarize, wasapi_exclusive_settings)

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = ["mode", "latency", "block_period_ms", "driver_latency_ms",
           "driver_overhead_ms", "overflows", "p99_us", "max_us", "verdict"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--samplerate", type=int, default=48_000)
    p.add_argument("--blocksize", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--hostapi", default=None)
    p.add_argument("--repeats", type=int, default=1,
                   help="runs per cell; use 3 if you want to claim anything "
                        "about callback timing differences between cells")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = int(args.device) if (args.device or "").isdigit() else args.device

    if args.hostapi:
        found = find_input_device(args.hostapi)
        if found is None:
            raise SystemExit(f"no input device for host API {args.hostapi!r}")
        device = found
        print(f"host API {args.hostapi!r} -> device index {device}\n")

    cells = [("shared", False), ("exclusive", True)]
    latencies = ["high", "low"]

    rows = []
    for mode, exclusive in cells:
        extra = wasapi_exclusive_settings() if exclusive else None
        if exclusive and extra is None:
            print(f"skipping {mode}: WASAPI exclusive unavailable here")
            continue
        for lat in latencies:
            for rep in range(args.repeats):
                label = f"{mode}/{lat}" + (f" #{rep+1}" if args.repeats > 1 else "")
                print(f"--- {label} ...", flush=True)
                try:
                    res = capture(duration_s=args.duration,
                                  samplerate=args.samplerate,
                                  blocksize=args.blocksize, device=device,
                                  extra_settings=extra, latency=lat)
                except Exception as exc:
                    print(f"    FAILED: {exc}")
                    rows.append({c: "" for c in COLUMNS}
                                | {"mode": mode, "latency": lat,
                                   "verdict": f"failed: {exc}"})
                    continue

                s = summarize(res["durations_ns"], scale="us")
                period = res["block_period_ms"]
                overhead = res["driver_latency_ms"] - period
                max_pct = s["max"] / (period * 1000.0) * 100

                verdict = ("DROPOUTS" if res["input_overflows"]
                           else "fragile" if max_pct > 50
                           else "tight" if max_pct > 25
                           else "clean")

                rows.append({
                    "mode": mode,
                    "latency": lat,
                    "block_period_ms": round(period, 3),
                    "driver_latency_ms": round(res["driver_latency_ms"], 3),
                    "driver_overhead_ms": round(overhead, 3),
                    "overflows": res["input_overflows"],
                    "p99_us": round(s["p99"], 1),
                    "max_us": round(s["max"], 1),
                    "verdict": verdict,
                })
                print(f"    input path {res['driver_latency_ms']:.2f} ms "
                      f"(buffer {period:.2f} + driver {overhead:.2f})  "
                      f"-> {verdict}")

    print("\n" + markdown_table(rows, COLUMNS))

    valid = [r for r in rows if isinstance(r.get("driver_latency_ms"), float)]
    if len(valid) >= 2:
        worst = max(valid, key=lambda r: r["driver_latency_ms"])
        best = min(valid, key=lambda r: r["driver_latency_ms"])
        saved = worst["driver_latency_ms"] - best["driver_latency_ms"]
        print(f"\nworst: {worst['mode']}/{worst['latency']} "
              f"{worst['driver_latency_ms']:.2f} ms")
        print(f"best:  {best['mode']}/{best['latency']} "
              f"{best['driver_latency_ms']:.2f} ms")
        print(f"free latency available through configuration: {saved:.2f} ms")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_csv(RESULTS / f"day1-matrix-bs{args.blocksize}-{stamp}.csv", rows)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()