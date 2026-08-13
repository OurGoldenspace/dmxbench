"""Art-Net (ArtDMX) output.

Art-Net carries DMX512 over UDP. This module builds ArtDMX packets and
sends them, doing no allocation on the hot path.

Packet layout (Art-Net 4 spec, ArtDmx / OpOutput):

    offset  size  field
    0       8     'A','r','t','-','N','e','t',0
    8       2     OpCode = 0x5000              LITTLE endian
    10      2     ProtVerHi/Lo = 14            BIG endian
    12      1     Sequence  (1..255, 0 = sequencing disabled)
    13      1     Physical  (informational only)
    14      1     SubUni    (low byte of universe address)
    15      1     Net       (high 7 bits of universe address)
    16      2     Length    (BIG endian, EVEN, 2..512)
    18      N     channel data

Note the endianness is genuinely inconsistent between OpCode and the
other 16-bit fields. That is the spec, not a bug here.

What this module does NOT tell you: how long the frame takes on the
physical DMX wire. That is set by the gateway, and the arithmetic for it
lives in `dmx_wire_time_us` below for comparison purposes only.
"""

from __future__ import annotations

import socket
import time

import numpy as np

ARTNET_PORT = 6454
ARTNET_ID = b"Art-Net\x00"
OP_DMX = 0x5000
PROT_VER = 14
HEADER_LEN = 18
MAX_CHANNELS = 512

# --- DMX512 physical timing constants -------------------------------------
# 250 kbaud, 11 bits per slot (1 start + 8 data + 2 stop) = 44 us per slot.
SLOT_US = 44.0
BREAK_US = 88.0          # spec minimum
MAB_US = 8.0             # mark after break, spec minimum
BREAK_MAB_US = BREAK_US + MAB_US


def dmx_wire_time_us(channels: int) -> float:
    """Theoretical DMX512 frame time for a universe of `channels` slots.

    DERIVED FROM THE SPEC, NOT MEASURED. Whether a given gateway actually
    shortens its frame when sent fewer channels — rather than padding back
    out to 512 — is an empirical question about that gateway's firmware,
    and is the thing week 3 tests with a photodiode.

    The +1 accounts for the start code slot, which always transmits.
    """
    return BREAK_MAB_US + SLOT_US * (channels + 1)


def dmx_max_refresh_hz(channels: int) -> float:
    """Upper bound on refresh rate for a universe of `channels` slots."""
    return 1_000_000.0 / dmx_wire_time_us(channels)


class ArtNetSender:
    """Sends ArtDMX packets with no per-frame allocation.

        tx = ArtNetSender("127.0.0.1", universe=0, channels=24)
        tx.set_channel(0, 255)
        elapsed_ns = tx.send()

    The packet buffer and socket are created once. `send()` overwrites a
    slice of the existing bytearray and returns its own elapsed time in
    nanoseconds, so callers do no timing arithmetic of their own.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = ARTNET_PORT,
        universe: int = 0,
        channels: int = MAX_CHANNELS,
        physical: int = 0,
    ) -> None:
        if not 1 <= channels <= MAX_CHANNELS:
            raise ValueError(f"channels must be 1..{MAX_CHANNELS}, got {channels}")

        self.host = host
        self.port = port
        self.universe = universe
        self.physical = physical
        self._sequence = 1
        # Counts ICMP port-unreachable replies. Nonzero simply means
        # nothing is listening — expected when benchmarking the send path.
        self.unreachable = 0

        # Full-size buffer allocated once. Short universes send a slice of it.
        self._buf = bytearray(HEADER_LEN + MAX_CHANNELS)
        self._buf[0:8] = ARTNET_ID
        self._buf[8] = OP_DMX & 0xFF            # little endian OpCode
        self._buf[9] = (OP_DMX >> 8) & 0xFF
        self._buf[10] = (PROT_VER >> 8) & 0xFF  # big endian ProtVer
        self._buf[11] = PROT_VER & 0xFF
        self._buf[13] = physical & 0xFF
        self._buf[14] = universe & 0xFF         # SubUni
        self._buf[15] = (universe >> 8) & 0x7F  # Net

        self._view = memoryview(self._buf)
        self.set_channels(channels)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # connect() on a UDP socket fixes the peer, so send() skips address
        # resolution on every call. Measurably cheaper than sendto().
        self._sock.connect((host, port))

    # -- configuration ----------------------------------------------------

    def set_channels(self, channels: int) -> None:
        """Resize the universe. Length must be even per the spec."""
        if not 1 <= channels <= MAX_CHANNELS:
            raise ValueError(f"channels must be 1..{MAX_CHANNELS}, got {channels}")
        length = channels + (channels & 1)      # round up to even
        self.channels = length
        self._buf[16] = (length >> 8) & 0xFF    # big endian Length
        self._buf[17] = length & 0xFF
        self._packet_len = HEADER_LEN + length

    def set_channel(self, index: int, value: int) -> None:
        self._buf[HEADER_LEN + index] = value & 0xFF

    def set_data(self, data) -> None:
        """Copy channel data into the packet buffer without allocating."""
        if isinstance(data, np.ndarray):
            data = data.astype(np.uint8, copy=False).tobytes()
        n = min(len(data), self.channels)
        self._buf[HEADER_LEN:HEADER_LEN + n] = data[:n]

    def blackout(self) -> None:
        for i in range(HEADER_LEN, HEADER_LEN + self.channels):
            self._buf[i] = 0

    # -- hot path ---------------------------------------------------------

    def send(self, data=None) -> int:
        """Send one ArtDMX packet. Returns elapsed nanoseconds.

        A connected UDP socket surfaces ICMP port-unreachable from the
        previous send as an error on the next one: ConnectionRefusedError
        on Linux, WSAECONNRESET on Windows. That happens whenever nothing
        is listening, which is normal when benchmarking the send path
        alone. It is counted, not raised.
        """
        if data is not None:
            self.set_data(data)

        # Sequence wraps 1..255; 0 means sequencing disabled.
        self._buf[12] = self._sequence
        self._sequence = self._sequence + 1 if self._sequence < 255 else 1

        t0 = time.perf_counter_ns()
        try:
            self._sock.send(self._view[:self._packet_len])
        except (ConnectionRefusedError, ConnectionResetError):
            self.unreachable += 1
        return time.perf_counter_ns() - t0

    # -- introspection ----------------------------------------------------

    @property
    def packet_bytes(self) -> int:
        return self._packet_len

    def header_hex(self) -> str:
        return self._buf[:HEADER_LEN].hex(" ")

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "ArtNetSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"ArtNetSender({self.host}:{self.port} universe={self.universe} "
                f"channels={self.channels} packet={self._packet_len}B)")


def parse_artdmx(packet: bytes) -> dict | None:
    """Decode an ArtDMX packet. Returns None if it is not one.

    Used by the listener to verify that what we send is well formed.
    """
    if len(packet) < HEADER_LEN or packet[0:8] != ARTNET_ID:
        return None
    opcode = packet[8] | (packet[9] << 8)
    if opcode != OP_DMX:
        return {"opcode": opcode, "artdmx": False}
    length = (packet[16] << 8) | packet[17]
    return {
        "artdmx": True,
        "opcode": opcode,
        "protver": (packet[10] << 8) | packet[11],
        "sequence": packet[12],
        "physical": packet[13],
        "universe": packet[14] | ((packet[15] & 0x7F) << 8),
        "length": length,
        "payload_len": len(packet) - HEADER_LEN,
        "packet_len": len(packet),
        "data": packet[HEADER_LEN:HEADER_LEN + length],
    }