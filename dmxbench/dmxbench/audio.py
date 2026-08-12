"""Audio device discovery and capture.

Everything about the capture path that is not measurement lives here.
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from .timing import buffer_latency_ms

DEFAULT_SAMPLERATE = 48_000
DEFAULT_BLOCKSIZE = 512


def list_devices() -> str:
    """Human-readable device list. Run this before anything else."""
    return str(sd.query_devices())


def default_input_info() -> dict:
    d = sd.query_devices(kind="input")
    apis = sd.query_hostapis()
    return {
        "name": d["name"],
        "index": d.get("index"),
        "hostapi": apis[d["hostapi"]]["name"],
        "max_input_channels": d["max_input_channels"],
        "default_samplerate": d["default_samplerate"],
        "default_low_input_latency_ms": d["default_low_input_latency"] * 1000,
        "default_high_input_latency_ms": d["default_high_input_latency"] * 1000,
    }


def input_devices_by_hostapi() -> dict[str, list[dict]]:
    """Group every input-capable device by its host API.

    On Windows this matters enormously. The default host API is usually
    MME, a 1991 interface whose driver-level latency alone is tens of
    milliseconds — larger than the entire budget this project measures.
    WASAPI or ASIO are the ones worth using.
    """
    apis = sd.query_hostapis()
    grouped: dict[str, list[dict]] = {a["name"]: [] for a in apis}
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            grouped[apis[d["hostapi"]]["name"]].append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "default_samplerate": d["default_samplerate"],
                "low_latency_ms": d["default_low_input_latency"] * 1000,
                "high_latency_ms": d["default_high_input_latency"] * 1000,
            })
    return {k: v for k, v in grouped.items() if v}


def find_input_device(hostapi_substr: str) -> int | None:
    """Return the default input device index for a host API, matched by name.

        find_input_device("WASAPI")   # Windows, low latency
        find_input_device("ASIO")     # Windows, lowest latency
        find_input_device("Core")     # macOS CoreAudio
    """
    for api in sd.query_hostapis():
        if hostapi_substr.lower() in api["name"].lower():
            idx = api["default_input_device"]
            return idx if idx >= 0 else None
    return None


def wasapi_exclusive_settings():
    """Extra settings for WASAPI exclusive mode, or None if unavailable.

    Shared mode routes through the Windows audio engine, which adds its
    own buffering. Exclusive mode hands the device to this process alone
    and typically cuts input latency substantially.
    """
    try:
        return sd.WasapiSettings(exclusive=True)
    except AttributeError:
        return None


def capture(
    duration_s: float = 60.0,
    samplerate: int = DEFAULT_SAMPLERATE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    device: int | str | None = None,
    channels: int = 1,
    extra_settings=None,
    latency: str | float = "low",
) -> dict:
    """Run an input stream and measure the callback, doing bounded work only.

    The callback does exactly three things: read the clock, compute one
    RMS value over the block, and record the elapsed time. No printing,
    no allocation, no I/O — those would block the audio thread and cause
    a dropout.

    Args:
        latency: "low", "high", or a value in seconds. sounddevice
            defaults to "high", which on Windows WASAPI means the
            device's high-latency mode — often 5x the low setting.
            This project defaults to "low" deliberately.

    Returns a dict with the raw per-callback durations plus stream metadata.
    """
    # Pre-allocate. Growing a list inside the callback can trigger a
    # reallocation at an unpredictable moment, which is exactly the kind
    # of unbounded work the audio thread must never do.
    est = int(duration_s * samplerate / blocksize)
    capacity = est + 1024

    durations_ns = np.zeros(capacity, dtype=np.int64)
    rms = np.zeros(capacity, dtype=np.float32)

    # A one-element list is a cheap mutable counter usable from the callback.
    n = [0]
    status_events = [0]
    input_overflows = [0]

    def callback(indata, frames, time_info, status):
        t0 = time.perf_counter_ns()

        if status:
            status_events[0] += 1
            if status.input_overflow:
                input_overflows[0] += 1

        i = n[0]
        if i < capacity:
            block = indata[:, 0]
            rms[i] = np.sqrt(np.mean(block * block))
            durations_ns[i] = time.perf_counter_ns() - t0
            n[0] = i + 1

    t_wall_start = time.perf_counter()
    with sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        device=device,
        dtype="float32",
        callback=callback,
        extra_settings=extra_settings,
        latency=latency,
    ) as stream:
        # The driver's own estimate of input latency. This sits on top of
        # the block period and is where host API and latency setting show up.
        driver_latency_ms = stream.latency * 1000
        api_name = sd.query_hostapis(
            sd.query_devices(stream.device)["hostapi"])["name"]
        time.sleep(duration_s)
    t_wall = time.perf_counter() - t_wall_start

    count = n[0]
    # Expected count is derived from the REQUESTED duration, not wall time.
    # Wall time includes stream open/close, which WASAPI shared mode can
    # make slow enough to look like hundreds of missing callbacks.
    expected = int(duration_s * samplerate / blocksize)

    return {
        "durations_ns": durations_ns[:count],
        "rms": rms[:count],
        "callbacks": count,
        "expected_callbacks": expected,
        "status_events": status_events[0],
        "input_overflows": input_overflows[0],
        "samplerate": samplerate,
        "blocksize": blocksize,
        "channels": channels,
        "wall_time_s": t_wall,
        "block_period_ms": buffer_latency_ms(blocksize, samplerate),
        "driver_latency_ms": driver_latency_ms,
        "hostapi": api_name,
        "latency_setting": str(latency),
        "exclusive": extra_settings is not None,
    }