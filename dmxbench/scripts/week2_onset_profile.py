#!/usr/bin/env python3
"""Week 2: does onset detection fit in the callback budget?

    python scripts/week2_onset_profile.py
    python scripts/week2_onset_profile.py --blocksizes 128 256 512 --history 4096 8192

Feeds synthetic audio through each detector block by block, exactly as a
callback would, and times the per-block cost against the block period.

No audio hardware needed — this is a pure compute benchmark, so it
isolates detector cost from driver and scheduling effects. The live
version (week2_live_chain.py) adds those back in.

The verdict column uses the same rule as the day 1 sweep: judged on MAX,
because one overrun is an audible click.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from dmxbench import markdown_table, save_csv, summarize
from dmxbench.onset import DETECTORS

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = ["detector", "blocksize", "history", "n_fft", "period_ms",
           "p50_us", "p95_us", "p99_us", "max_us", "pct_period_max",
           "onsets", "onsets_per_sec", "verdict"]


def synth_audio(seconds: float, sr: int, bpm: float = 128.0,
                seed: int = 0) -> np.ndarray:
    """Kick-and-hat pattern over noise. Sharp transients on a known grid."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = rng.normal(0, 0.01, n).astype(np.float32)

    beat = 60.0 / bpm
    t = np.arange(n) / sr

    for i in range(int(seconds / beat)):
        start = int(i * beat * sr)
        # kick: 60 Hz sine with a fast exponential decay
        dur = int(0.12 * sr)
        end = min(start + dur, n)
        env = np.exp(-np.arange(end - start) / (0.03 * sr))
        y[start:end] += (0.8 * env *
                         np.sin(2 * np.pi * 60 * t[:end - start])).astype(np.float32)

        # hat on the offbeat: filtered noise burst
        h = start + int(beat * sr / 2)
        hend = min(h + int(0.03 * sr), n)
        if hend > h:
            henv = np.exp(-np.arange(hend - h) / (0.005 * sr))
            y[h:hend] += (0.3 * henv *
                          rng.normal(0, 1, hend - h)).astype(np.float32)

    return np.clip(y, -1.0, 1.0)


def profile(detector_name: str, y: np.ndarray, sr: int, blocksize: int,
            history: int, n_fft: int, hop: int, warmup: int = 50) -> dict:
    cls = DETECTORS[detector_name]
    kw = dict(samplerate=sr, n_fft=n_fft, hop=hop)
    if detector_name == "librosa":
        kw["history"] = history
    det = cls(**kw)

    nblocks = y.size // blocksize

    # Warm up before timing. librosa lazily imports and JIT-compiles on
    # first call; without this the first block reads ~1.8 SECONDS and
    # poisons the max column with a number that is measuring numba, not
    # onset detection.
    import time
    for i in range(min(warmup, nblocks)):
        det.process(y[i * blocksize:(i + 1) * blocksize])

    durations = np.zeros(nblocks, dtype=np.int64)
    onsets = 0

    for i in range(nblocks):
        block = y[i * blocksize:(i + 1) * blocksize]
        t0 = time.perf_counter_ns()
        hit = det.process(block)
        durations[i] = time.perf_counter_ns() - t0
        onsets += bool(hit)

    return {"durations_ns": durations, "onsets": onsets, "blocks": nblocks}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samplerate", type=int, default=48_000)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--blocksizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--history", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--detectors", nargs="+",
                   default=["librosa", "incremental"])
    p.add_argument("--warmup", type=int, default=50,
                   help="blocks discarded before timing (librosa JIT)")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sr = args.samplerate

    print(f"synthesising {args.seconds:.0f}s of test audio @ {sr} Hz ...")
    y = synth_audio(args.seconds, sr)

    rows = []
    for name in args.detectors:
        # history only affects the librosa detector; run it once otherwise
        histories = args.history if name == "librosa" else [0]
        for hist in histories:
            for bs in args.blocksizes:
                period_ms = bs / sr * 1000.0
                if name == "librosa" and hist < args.n_fft:
                    print(f"  skip {name} history={hist} < n_fft={args.n_fft}")
                    continue

                res = profile(name, y, sr, bs, hist, args.n_fft, hop=bs,
                              warmup=args.warmup)
                s = summarize(res["durations_ns"], scale="us")
                pct = s["max"] / 1000.0 / period_ms * 100

                verdict = ("BLOWN" if pct >= 100
                           else "fragile" if pct > 50
                           else "tight" if pct > 25
                           else "clean")

                rows.append({
                    "detector": name,
                    "blocksize": bs,
                    "history": hist,
                    "n_fft": args.n_fft,
                    "period_ms": round(period_ms, 3),
                    "p50_us": round(s["p50"], 1),
                    "p95_us": round(s["p95"], 1),
                    "p99_us": round(s["p99"], 1),
                    "max_us": round(s["max"], 1),
                    "pct_period_max": round(pct, 1),
                    "onsets": res["onsets"],
                    "onsets_per_sec": round(res["onsets"] / args.seconds, 2),
                    "verdict": verdict,
                })
                print(f"  {name:<12} bs={bs:<5} hist={hist:<6} "
                      f"period={period_ms:6.2f}ms  "
                      f"p50={s['p50']:>8.1f}us p99={s['p99']:>8.1f}us "
                      f"max={s['max']:>8.1f}us ({pct:5.1f}%)  "
                      f"onsets={res['onsets']:<4} -> {verdict}")

    print("\n" + markdown_table(rows, COLUMNS))

    lib = [r for r in rows if r["detector"] == "librosa"]
    inc = [r for r in rows if r["detector"] == "incremental"]
    if lib and inc:
        for bs in sorted({r["blocksize"] for r in inc}):
            l = [r for r in lib if r["blocksize"] == bs]
            i = [r for r in inc if r["blocksize"] == bs]
            if l and i:
                speedup = max(x["p50_us"] for x in l) / i[0]["p50_us"]
                print(f"\nblocksize {bs}: incremental is {speedup:.0f}x cheaper "
                      f"at p50 than the worst librosa config")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    path = save_csv(RESULTS / f"week2-onset{tag}-{stamp}.csv", rows)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()