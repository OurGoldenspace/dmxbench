"""Onset detection, in two implementations that produce comparable output.

An onset is a moment where spectral energy rises sharply — a kick, a
snare, a strum. Detecting one is three steps:

  1. STFT      window the signal, FFT each window, take magnitudes
  2. flux      per bin, compare to the previous frame and sum only the
               INCREASES (half-wave rectification: energy falling does
               not signal a new event)
  3. peak pick local maximum above an adaptive threshold

Two detectors implement this:

  LibrosaOnsetDetector    calls librosa.onset.onset_strength on a whole
                          history window every callback. Correct,
                          idiomatic, and recomputes almost everything it
                          computed last time.

  IncrementalOnsetDetector keeps one rfft per hop and one previous
                          magnitude spectrum. Constant work per callback,
                          independent of history length.

Both exist so the cost of the naive version can be MEASURED rather than
asserted. Do not delete the slow one.
"""

from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError:  # librosa is only needed for the naive detector
    librosa = None


# --- shared helpers -------------------------------------------------------

def hann(n: int) -> np.ndarray:
    """Periodic Hann window, matching what librosa/scipy use for STFT."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


class AdaptivePeakPicker:
    """Flags a flux value as an onset if it stands out from recent history.

    Threshold is median + delta * MAD over a sliding window, with a
    refractory period so one transient does not fire several frames in a
    row. Median and MAD rather than mean and std because a loud kick
    should not raise the bar against itself.
    """

    def __init__(self, history: int = 32, delta: float = 3.0,
                 refractory_ms: float = 40.0, frame_ms: float = 5.33,
                 floor: float = 1e-6):
        self.buf = np.zeros(history, dtype=np.float32)
        self.n = 0
        self.delta = delta
        # Refractory must be expressed in TIME, not frames. A fixed frame
        # count means the deadband shrinks as blocksize shrinks, so the
        # same signal yields more onsets at smaller blocksizes — an
        # artefact of the buffer size, not a property of the music.
        self.refractory_ms = refractory_ms
        self.frame_ms = frame_ms
        self.refractory = max(1, int(round(refractory_ms / frame_ms)))
        self.since_onset = self.refractory
        self.floor = floor
        self.last_threshold = 0.0

    def __call__(self, flux: float) -> bool:
        i = self.n % self.buf.size
        filled = min(self.n, self.buf.size)

        if filled < 8:                       # not enough history yet
            self.buf[i] = flux
            self.n += 1
            self.since_onset += 1
            return False

        window = self.buf[:filled]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med))) + self.floor
        threshold = med + self.delta * mad
        self.last_threshold = threshold

        is_onset = flux > threshold and self.since_onset >= self.refractory
        self.since_onset = 0 if is_onset else self.since_onset + 1

        self.buf[i] = flux
        self.n += 1
        return is_onset


# --- detector 1: naive, recompute everything ------------------------------

class LibrosaOnsetDetector:
    """Recomputes the onset envelope over the whole history each call.

    This is what the obvious implementation looks like, and it is what
    week 1's day-3 sketch did. Its cost grows with history length even
    though only one hop of new audio arrived.
    """

    name = "librosa"

    def __init__(self, samplerate: int = 48_000, n_fft: int = 2048,
                 hop: int = 256, history: int = 8192, **picker_kw):
        if librosa is None:
            raise ImportError("librosa is required for LibrosaOnsetDetector")
        self.sr = samplerate
        self.n_fft = n_fft
        self.hop = hop
        self.ring = np.zeros(history, dtype=np.float32)
        picker_kw.setdefault("frame_ms", hop / samplerate * 1000.0)
        self.picker = AdaptivePeakPicker(**picker_kw)
        self.flux = 0.0

    def process(self, block: np.ndarray) -> bool:
        n = block.size
        self.ring[:-n] = self.ring[n:]        # shift, cheaper than np.roll
        self.ring[-n:] = block

        env = librosa.onset.onset_strength(
            y=self.ring, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop,
            center=False)
        self.flux = float(env[-1]) if env.size else 0.0
        return self.picker(self.flux)


# --- detector 2: incremental, constant work -------------------------------

class IncrementalOnsetDetector:
    """One rfft per hop, one previous magnitude spectrum retained.

    Cost is fixed by n_fft alone. History length does not enter into it,
    because nothing older than the current window is ever revisited.

    Flux is normalised by bin count so its scale is comparable to the
    librosa detector's, letting the two be plotted on one axis.
    """

    name = "incremental"

    def __init__(self, samplerate: int = 48_000, n_fft: int = 2048,
                 hop: int = 256, **picker_kw):
        self.sr = samplerate
        self.n_fft = n_fft
        self.hop = hop

        self.window = hann(n_fft).astype(np.float32)
        self.buf = np.zeros(n_fft, dtype=np.float32)      # newest n_fft samples
        self.prev_mag = np.zeros(n_fft // 2 + 1, dtype=np.float32)
        self._scratch = np.empty(n_fft, dtype=np.float32)

        picker_kw.setdefault("frame_ms", hop / samplerate * 1000.0)
        self.picker = AdaptivePeakPicker(**picker_kw)
        self.flux = 0.0
        self.frames = 0

    def process(self, block: np.ndarray) -> bool:
        n = block.size
        if n >= self.n_fft:
            self.buf[:] = block[-self.n_fft:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = block

        np.multiply(self.buf, self.window, out=self._scratch)
        mag = np.abs(np.fft.rfft(self._scratch)).astype(np.float32)

        # Half-wave rectified spectral flux: only increases count.
        diff = mag - self.prev_mag
        np.maximum(diff, 0.0, out=diff)
        self.flux = float(diff.sum()) / diff.size

        self.prev_mag = mag
        self.frames += 1
        return self.picker(self.flux)


# --- offline reference ----------------------------------------------------

def offline_onsets(y: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """Onset times in seconds, computed offline. Ground truth for agreement."""
    if librosa is None:
        raise ImportError("librosa is required for offline_onsets")
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                      hop_length=hop, units="time")


DETECTORS = {
    "librosa": LibrosaOnsetDetector,
    "incremental": IncrementalOnsetDetector,
}