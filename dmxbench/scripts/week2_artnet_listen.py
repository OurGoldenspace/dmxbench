#!/usr/bin/env python3
"""Listen for Art-Net packets and verify they are well formed.

    python scripts/week2_artnet_listen.py

Run this in one terminal, the bench in another. It exists so you can
prove your packets are correct before any gateway hardware is involved:
if the header, OpCode, universe and Length are right here, they are right
on the wire.

Prints the first few packets in full, then counts and reports rate.
"""

import argparse
import socket
import time

from dmxbench.artnet import ARTNET_PORT, parse_artdmx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=ARTNET_PORT)
    p.add_argument("--show", type=int, default=5,
                   help="how many packets to print in full")
    p.add_argument("--report-every", type=float, default=2.0,
                   help="seconds between rate reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)

    print(f"listening on {args.bind}:{args.port}  (ctrl-c to stop)\n")

    shown = 0
    total = 0
    bad = 0
    window = 0
    window_bytes = 0
    last_report = time.perf_counter()
    last_seq = None
    seq_gaps = 0

    try:
        while True:
            try:
                packet, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            info = parse_artdmx(packet)
            if info is None or not info.get("artdmx"):
                bad += 1
                continue

            total += 1
            window += 1
            window_bytes += len(packet)

            # Sequence should increment by 1, wrapping 255 -> 1.
            seq = info["sequence"]
            if last_seq is not None:
                expected = last_seq + 1 if last_seq < 255 else 1
                if seq != expected:
                    seq_gaps += 1
            last_seq = seq

            if shown < args.show:
                shown += 1
                data = info["data"]
                preview = " ".join(f"{b:02x}" for b in data[:16])
                print(f"packet #{shown} from {addr[0]}")
                print(f"  protver     {info['protver']}")
                print(f"  sequence    {info['sequence']}")
                print(f"  universe    {info['universe']}")
                print(f"  Length      {info['length']}   <- declared channel count")
                print(f"  payload     {info['payload_len']} bytes")
                print(f"  packet      {info['packet_len']} bytes "
                      f"(18 header + {info['payload_len']})")
                print(f"  data[0:16]  {preview}")
                ok = info["length"] == info["payload_len"]
                print(f"  Length matches payload: {ok}"
                      f"{'' if ok else '   <-- MALFORMED'}\n")

            now = time.perf_counter()
            if now - last_report >= args.report_every:
                dt = now - last_report
                print(f"{window / dt:8.1f} pkt/s  "
                      f"{window_bytes / dt / 1024:8.1f} KiB/s  "
                      f"total={total}  bad={bad}  seq_gaps={seq_gaps}")
                window = 0
                window_bytes = 0
                last_report = now

    except KeyboardInterrupt:
        print(f"\nstopped. artdmx={total} non-artdmx={bad} seq_gaps={seq_gaps}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()