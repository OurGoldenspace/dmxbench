# dmxbench

Measuring audio-to-photon latency for realtime AI lighting systems.

Music-reactive lighting engines advertise sub-100 ms audio-to-light
synchrony. That number is almost always quoted as a median, and almost
never measured end to end — from the transient in the audio to the
photons actually leaving the fixture.

This project measures it properly, reports the tail rather than the
median, and answers the question that follows from it: **how much
predictive lookahead does a lighting engine need, per fixture class, for
the light to land on the beat?**

A moving head cannot hit a beat reactively. Pan/tilt is mechanical —
motor acceleration, travel, settle. Even a zero-latency model arrives
late. So the useful output is not a latency figure, it is a lookahead
budget.

---

## Status

- [x] **Week 1 — audio capture path and its timing floor**
- [ ] Week 2 — DMX/Art-Net output, inference optimisation sweep
- [ ] Week 3 — photodiode loopback, fixture response, lookahead solver

---

## The latency budget

Filled in as the project progresses. Everything is p99 unless stated.

| Stage | Latency | How measured | Status |
|---|---|---|---|
| Driver / host API | ? | `stream.latency`, MME vs WASAPI | measured |
| Audio buffer fill | `blocksize / samplerate` | arithmetic, fixed by config | measured |
| Callback processing | ? | `perf_counter_ns` per stage | measured |
| Feature extraction | ? | per-stage instrumentation | week 1 |
| Model inference | ? | fp32 / fp16 / int8 sweep | week 2 |
| Decision logic | ? | per-stage instrumentation | week 1 |
| DMX512 frame time | ~23 ms | protocol limit, ~44 Hz refresh | week 2 |
| Fixture response (LED PAR) | ? | photodiode | week 3 |
| Fixture response (moving head) | ? | photodiode | week 3 |
| **End to end** | **?** | audio loopback cross-correlation | week 3 |

### Why p99 and not mean

A mean of 3 ms with a p99 of 180 ms means one beat in a hundred lands
visibly late. On a drop, that is the beat the room remembers. Every
number in this repo is a distribution.

---

## Install

Requires Python 3.10+.

### Windows

PortAudio ships inside the `sounddevice` wheel — nothing extra to install.

```powershell
tar -xzf dmxbench-week1.tar.gz
cd dmxbench

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell
# .venv\Scripts\activate.bat      # cmd.exe

pip install -e .
```

If PowerShell refuses to run the activate script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

That is a per-user setting and does not require admin.

### macOS / Linux

```bash
brew install portaudio            # macOS
sudo apt install libportaudio2    # Debian/Ubuntu

tar -xzf dmxbench-week1.tar.gz && cd dmxbench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e .` installs the package in editable mode, which is what
lets `scripts/` import `dmxbench/` without path hacks.

---

## The host API trap (Windows)

On Windows, `sounddevice` defaults to **MME**, an audio interface from
1991. Its driver-level buffering alone can be tens of milliseconds —
larger than the entire budget this project exists to measure. Measure on
MME and you will conclude the pipeline is hopeless when the problem is
the driver.

| Host API | Verdict |
|---|---|
| MME | avoid — legacy, high driver latency |
| DirectSound | avoid — better than MME, still buffered |
| **WASAPI** | **use this**, with `--exclusive` |
| WDM-KS | good, kernel streaming, can be finicky |
| **ASIO** | **best**, if your interface has ASIO drivers |

Shared-mode WASAPI routes through the Windows audio engine, which adds
its own buffering. Exclusive mode hands the device to this process alone.

```powershell
python scripts/day1_devices.py                          # see what exists
python scripts/day1_capture.py --hostapi WASAPI --exclusive
```

Running the same sweep on MME and on WASAPI-exclusive, and reporting both,
is itself a result worth putting in the README — it quantifies how much of
a "latency problem" is really a driver-configuration problem.

If you buy an audio interface for week 3, ASIO drivers are the reason to
prefer a Focusrite/MOTU/Behringer class device over a USB dongle.

---

## Week 1 — running it

```bash
# 0. What audio hardware is available, and on which host API?
python scripts/day1_devices.py

# 1. Sixty seconds of capture. Success = zero input overflows.
python scripts/day1_capture.py --device <index>
python scripts/day1_capture.py --hostapi WASAPI --exclusive   # Windows

# 2. Sweep the buffer size. This is the first real result.
python scripts/day1_sweep.py --hostapi WASAPI --exclusive
```

Play music through speakers and let the built-in mic hear it. A proper
line-level loopback rig arrives in week 3; for the capture-path timing
it makes no difference.

### What to look for

**The driver is a second floor.** `driver_latency_ms` is reported on top
of the block period and is set by the host API, not by your code. On
Windows this is the difference between MME and WASAPI-exclusive.

**Block period is your floor.** 512 samples at 48 kHz is 10.67 ms. The
first sample in that block was captured 10.67 ms before your code saw
it. Nothing you write recovers it. Before a single line of detection
code runs, a tenth of the 100 ms budget is gone.

**Headroom is your budget.** The callback must finish well inside the
block period or the sound card starves and you get a dropout. Whatever
is left over is the total budget for feature extraction, inference and
decision logic in weeks 2 and 3.

**Overflows are binary.** Zero is the only acceptable number. If you see
any, raise the blocksize and accept the latency, or reduce the work.

### Results

_Paste the sweep table here._

| blocksize | block_period_ms | overflows | p99_us | headroom_ms | verdict |
|---|---|---|---|---|---|
| | | | | | |

Smallest blocksize with `verdict=clean` is the floor for everything that
follows.

---

## Layout

```
dmxbench/
├── dmxbench/            package — reusable, no side effects on import
│   ├── __init__.py
│   ├── audio.py         device discovery + instrumented capture
│   └── timing.py        percentile reporting, CSV/JSON/markdown output
├── scripts/             runnable entry points, one per task
│   ├── day1_devices.py
│   ├── day1_capture.py
│   └── day1_sweep.py
├── results/             measurement output (gitignored except .gitkeep)
├── notebooks/           offline exploration, plots
├── pyproject.toml
└── README.md
```

The split matters: `scripts/` handles argument parsing, printing and file
paths; `dmxbench/` holds logic that returns data and prints nothing. Week
2 reuses the package unchanged.

---

## Design notes

**The callback never blocks.** No printing, no file I/O, no allocation
inside the audio callback. Arrays are pre-allocated before the stream
opens, because a list growing at an unpredictable moment is exactly the
unbounded work the audio thread cannot tolerate. The callback records; the
main thread analyses.

**Every stage is timed separately.** Not a total — a breakdown. You
cannot optimise what you have not attributed.

**Measurement before optimisation.** Week 1 deliberately uses a naive
feature extraction path. The point is to measure how wasteful it is; that
measurement is what justifies the week 2 rewrite.

---

## License

MIT
