#!/usr/bin/env python3
"""Week 2 day 1: how long does an Art-Net send take, and does size matter?

    python scripts/week2_artnet_bench.py
    python scripts/week2_artnet_bench.py --host 127.0.0.1 --sends 20000

Sweeps universe size and reports two very different numbers side by side:

  send_*_us       MEASURED. Time for this process to hand the packet to
                  the OS. Microseconds.

  wire_time_us    DERIVED from the DMX512 spec, not measured. What the
                  frame would take on a physical DMX line. Milliseconds.

The gap between them is the point: the software path is three orders of
magnitude cheaper than the protocol it feeds. Optimising the sender is
pointless; shortening the universe is where the latency is.

Whether a real gateway honours a short universe rather than padding back
to 512 is untested here. That is week 3, with a photodiode.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from dmxbench import markdown_table, print_summary, save_csv, summarize
from dmxbench.artnet import ArtNetSender, dmx_max_refresh_hz, dmx_wire_time_us

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = ["channels", "packet_bytes", "send_p50_us", "send_p95_us",
           "send_p99_us", "send_max_us", "wire_time_us", "max_refresh_hz",
           "wire_vs_send"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="where to send; a listener need not be running")
    p.add_argument("--port", type=int, default=6454)
    p.add_argument("--universe", type=int, default=0)
    p.add_argument("--sends", type=int, default=10_000, help="packets per size")
    p.add_argument("--warmup", type=int, default=500,
                   help="discarded sends before measuring")
    p.add_argument("--channels", type=int, nargs="+",
                   default=[8, 24, 64, 128, 256, 512])
    p.add_argument("--tag", default="")
    return p.parse_args()


def bench_one(host: str, port: int, universe: int, channels: int,
              sends: int, warmup: int) -> dict:
    """Time `sends` packets at one universe size."""
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 256, size=channels, dtype=np.uint8)

    with ArtNetSender(host=host, port=port, universe=universe,
                      channels=channels) as tx:
        tx.set_data(payload)

        for _ in range(warmup):
            tx.send()

        durations = np.zeros(sends, dtype=np.int64)
        for i in range(sends):
            durations[i] = tx.send()

        return {"durations_ns": durations, "packet_bytes": tx.packet_bytes,
                "channels": tx.channels, "header": tx.header_hex(),
                "unreachable": tx.unreachable}


def main() -> None:
    args = parse_args()

    print(f"sending to {args.host}:{args.port}, universe {args.universe}, "
          f"{args.sends} packets per size\n")

    rows = []
    unreachable_total = 0
    for ch in args.channels:
        res = bench_one(args.host, args.port, args.universe, ch,
                        args.sends, args.warmup)
        s = summarize(res["durations_ns"], scale="us")
        unreachable_total += res["unreachable"]
        wire = dmx_wire_time_us(res["channels"])
        refresh = dmx_max_refresh_hz(res["channels"])

        rows.append({
            "channels": res["channels"],
            "packet_bytes": res["packet_bytes"],
            "send_p50_us": round(s["p50"], 2),
            "send_p95_us": round(s["p95"], 2),
            "send_p99_us": round(s["p99"], 2),
            "send_max_us": round(s["max"], 2),
            "wire_time_us": round(wire, 1),
            "max_refresh_hz": round(refresh, 1),
            "wire_vs_send": round(wire / s["p50"], 1),
        })
        print(f"  {res['channels']:>4} ch  {res['packet_bytes']:>4} B  "
              f"send p50={s['p50']:>7.2f}us p99={s['p99']:>7.2f}us  |  "
              f"wire {wire / 1000:>6.2f}ms  max {refresh:>5.1f} Hz")

    print("\n" + markdown_table(rows, COLUMNS))

    if unreachable_total:
        print(f"\nnote: {unreachable_total} sends got ICMP port-unreachable — "
              "nothing is listening on that port.\n"
              "      Harmless for send-path timing. Run "
              "scripts/week2_artnet_listen.py to silence it.")

    full, short = rows[-1], rows[0]
    print(f"\nwire time {short['channels']} ch vs {full['channels']} ch: "
          f"{short['wire_time_us'] / 1000:.2f} ms vs "
          f"{full['wire_time_us'] / 1000:.2f} ms  "
          f"({full['wire_time_us'] / short['wire_time_us']:.1f}x)")
    print("  ^ DERIVED FROM SPEC. Not yet verified against real gateway firmware.")
    print(f"\nsoftware send is ~{full['wire_vs_send']:.0f}x cheaper than the wire "
          f"at 512 ch — the sender is not the bottleneck.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    path = save_csv(RESULTS / f"week2-artnet{tag}-{stamp}.csv", rows)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()