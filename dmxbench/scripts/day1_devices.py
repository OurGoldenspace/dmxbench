#!/usr/bin/env python3
"""Step 0: what audio hardware do we have, and on which host API?

    python scripts/day1_devices.py

The host API matters more than the device on Windows. The default is
usually MME, whose driver-level latency alone can exceed the entire
budget this project measures. Prefer WASAPI, or ASIO if you have it.

Note the index of the input you want, then:

    python scripts/day1_capture.py --device <index>
    python scripts/day1_capture.py --hostapi WASAPI --exclusive
"""

import sounddevice as sd

from dmxbench import default_input_info, input_devices_by_hostapi

# Rough guidance. Actual numbers come from your own sweep.
API_NOTES = {
    "MME": "avoid — legacy, high driver latency",
    "Windows DirectSound": "avoid — better than MME, still buffered",
    "Windows WASAPI": "GOOD — use this, add --exclusive",
    "Windows WDM-KS": "good — kernel streaming, can be finicky",
    "ASIO": "BEST — if you have an interface with ASIO drivers",
    "Core Audio": "GOOD — macOS default is already low latency",
    "ALSA": "GOOD — Linux, avoid the pulse/ device if you can",
}


def note_for(api_name: str) -> str:
    for key, note in API_NOTES.items():
        if key.lower() in api_name.lower():
            return note
    return ""


def main() -> None:
    print("=== input devices by host API ===\n")
    for api, devices in input_devices_by_hostapi().items():
        note = note_for(api)
        print(f"{api}" + (f"   [{note}]" if note else ""))
        for d in devices:
            print(f"    [{d['index']:>2}] {d['name'][:48]:<48} "
                  f"{d['channels']}ch  {d['default_samplerate']:>7.0f} Hz  "
                  f"driver low-latency {d['low_latency_ms']:>6.1f} ms")
        print()

    print("=== current default input ===")
    for k, v in default_input_info().items():
        print(f"  {k:32s} {v}")

    print("\n=== host API defaults ===")
    for i, api in enumerate(sd.query_hostapis()):
        print(f"  [{i}] {api['name']:<22} default input device: "
              f"{api['default_input_device']}")

    print(
        "\nPick a WASAPI or ASIO device if you are on Windows. The driver\n"
        "low-latency figure above is ON TOP OF the block period, and it is\n"
        "where MME quietly costs you 30-100 ms."
    )


if __name__ == "__main__":
    main()
