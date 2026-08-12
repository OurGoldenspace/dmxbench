"""dmxbench — measuring audio-to-photon latency for AI lighting systems.

Week 1 scope: the audio capture path and its timing floor.

The audio submodule is imported lazily. `sounddevice` raises at import
time if PortAudio is missing, and that should not stop you from using
the timing and reporting helpers on a machine without audio hardware.
"""

__version__ = "0.1.0"

from .timing import (
    buffer_latency_ms,
    markdown_table,
    print_summary,
    save_csv,
    save_json,
    summarize,
)

_AUDIO_NAMES = {
    "capture",
    "list_devices",
    "default_input_info",
    "input_devices_by_hostapi",
    "find_input_device",
    "wasapi_exclusive_settings",
}

_PORTAUDIO_HELP = (
    "PortAudio is required for audio capture but was not found.\n"
    "  macOS:         brew install portaudio\n"
    "  Debian/Ubuntu: sudo apt install libportaudio2\n"
    "then reinstall sounddevice: pip install --force-reinstall sounddevice"
)


def __getattr__(name):
    """Import the audio layer on first use, with a readable error if it fails."""
    if name in _AUDIO_NAMES:
        try:
            from . import audio
        except OSError as exc:
            raise OSError(f"{exc}\n\n{_PORTAUDIO_HELP}") from exc
        return getattr(audio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "capture",
    "list_devices",
    "default_input_info",
    "input_devices_by_hostapi",
    "find_input_device",
    "wasapi_exclusive_settings",
    "summarize",
    "print_summary",
    "buffer_latency_ms",
    "save_json",
    "save_csv",
    "markdown_table",
]
