#!/usr/bin/env python3
"""Day 1: prove we can capture audio without dropouts, and find the floor.

    python scripts/day1_capture.py
    python scripts/day1_capture.py --hostapi WASAPI --exclusive --latency low
    python scripts/day1_capture.py --device 9 --blocksize 256 --duration 20

Success criterion: input_overflows == 0, and the callback MAX is a small
fraction of the block period. That headroom is the entire budget for
detection and inference in weeks 2 and 3.
"""

import argparse
from datetime import datetime
from pathlib import Path

from dmxbench import (capture, find_input_device, print_summary, save_json,
                      summarize, wasapi_exclusive_settings)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    p.add_argument("--samplerate", type=int, default=48_000)
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--device", default=None,
                   help="device index or name (see day1_devices.py)")
    p.add_argument("--hostapi", default=None,
                   help="pick the default device of a host API, e.g. WASAPI, ASIO")
    p.add_argument("--exclusive", action="store_true",
                   help="WASAPI exclusive mode (Windows) — bypasses the audio engine")
    p.add_argument("--latency", default="low",
                   help="'low', 'high', or seconds. sounddevice's own default is "
                        "'high'; this project defaults to 'low'")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--tag", default="", help="label for the saved result file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = int(args.device) if (args.device or "").isdigit() else args.device

    if args.hostapi:
        found = find_input_device(args.hostapi)
        if found is None:
            raise SystemExit(f"no input device found for host API {args.hostapi!r}; "
                             "run scripts/day1_devices.py to see what exists")
        device = found
        print(f"host API {args.hostapi!r} -> device index {device}")

    extra = wasapi_exclusive_settings() if args.exclusive else None
    if args.exclusive and extra is None:
        print("warning: WASAPI exclusive mode unavailable on this platform")

    latency = float(args.latency) if _is_number(args.latency) else args.latency

    print(f"capturing {args.duration:.0f}s @ {args.samplerate} Hz, "
          f"blocksize={args.blocksize}, latency={latency!r}, "
          f"exclusive={bool(extra)} ...")

    res = capture(
        duration_s=args.duration,
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        device=device,
        channels=args.channels,
        extra_settings=extra,
        latency=latency,
    )

    summary = summarize(res["durations_ns"], scale="us")
    period_ms = res["block_period_ms"]
    input_path_ms = res["driver_latency_ms"]

    print("\n=== stream ===")
    print(f"  host API           {res['hostapi']}")
    print(f"  latency setting    {res['latency_setting']}   "
          f"exclusive={res['exclusive']}")
    print(f"  driver latency     {input_path_ms:.3f} ms   "
          f"<- the whole input path as the driver reports it")
    print(f"  block period       {period_ms:.3f} ms   "
          f"<- of which this much is your buffer")
    print(f"  driver overhead    {input_path_ms - period_ms:.3f} ms   "
          f"<- host API + latency setting, not your code")
    print(f"  callbacks          {res['callbacks']} "
          f"(expected ~{res['expected_callbacks']})")
    print(f"  status events      {res['status_events']}")
    print(f"  input overflows    {res['input_overflows']}"
          f"{'   <-- DROPOUTS, raise blocksize' if res['input_overflows'] else '   (clean)'}")

    print("\n=== callback duration ===")
    print_summary("callback", summary)

    max_pct = summary["max"] / 1000.0 / period_ms * 100
    p99_pct = summary["p99"] / 1000.0 / period_ms * 100
    headroom = period_ms - summary["max"] / 1000.0
    print(f"\n  p99 uses {p99_pct:.1f}% of the block period")
    print(f"  MAX uses {max_pct:.1f}% of the block period   "
          f"<- judge on this, one overrun is an audible click")
    print(f"  {headroom:.2f} ms worst-case headroom for detection + inference")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    excl = "-excl" if args.exclusive else ""
    lat = f"-{res['latency_setting']}"
    tag = f"-{args.tag}" if args.tag else ""
    name = f"day1-bs{args.blocksize}-sr{args.samplerate}{excl}{lat}{tag}-{stamp}.json"
    path = save_json(RESULTS / name, {
        "kind": "day1_capture",
        "timestamp": stamp,
        "config": {
            "samplerate": args.samplerate,
            "blocksize": args.blocksize,
            "duration_s": args.duration,
            "device": str(device),
            "channels": args.channels,
            "latency": res["latency_setting"],
            "exclusive": res["exclusive"],
        },
        "stream": {k: res[k] for k in (
            "callbacks", "expected_callbacks", "status_events",
            "input_overflows", "wall_time_s", "block_period_ms",
            "driver_latency_ms", "hostapi", "latency_setting", "exclusive")},
        "callback_us": summary,
        "derived": {
            "p99_pct_period": round(p99_pct, 2),
            "max_pct_period": round(max_pct, 2),
            "worst_case_headroom_ms": round(headroom, 3),
        },
    })
    print(f"\nsaved -> {path}")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()