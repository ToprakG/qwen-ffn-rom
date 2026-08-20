#!/usr/bin/env python3
"""Host for the 16-head D=4 FPGA farm over UART.

  python scripts/fpga_host.py --port /dev/ttyUSB1 --n 2000
  python scripts/fpga_host.py --dry-run

Prints wall-clock model tok/s. Cycle-implied tok/s needs --clk-hz.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC_TX = 0xA5
MAGIC_RX = 0x5A
REQ = 224
RSP = 196
N_HEADS = 16
D = 4


def pack_token(rng) -> bytes:
    buf = bytearray()
    for _ in range(3):
        buf.extend(int(x) & 0xFF for x in rng.integers(-128, 128, size=N_HEADS * D))
    buf.extend(int(x) & 0xFF for x in rng.integers(0, 256, size=N_HEADS))
    buf.extend(int(x) & 0xFF for x in rng.integers(0, 256, size=N_HEADS))
    assert len(buf) == REQ
    return bytes(buf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=os.environ.get("FPGA_PORT", "/dev/ttyUSB1"))
    p.add_argument("--baud", type=int, default=3_000_000)
    p.add_argument("--n", type=int, default=1000, help="tokens to send")
    p.add_argument("--clk-hz", type=float, default=100e6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(
        f"protocol  tx {1+REQ} B  rx {1+RSP} B  baud={args.baud}  "
        f"16 heads D=4  24 layers"
    )
    if args.dry_run:
        bits = (1 + REQ + 1 + RSP) * 10
        t = bits / args.baud
        print(f"uart floor  {1/t:.0f} tok/s  ({t*1e3:.2f} ms/token on the wire)")
        print("dry-run: no serial")
        return

    try:
        import serial  # type: ignore
    except ImportError as e:
        raise SystemExit("pip install pyserial") from e

    rng = __import__("numpy").random.default_rng(args.seed)
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    ser.reset_input_buffer()
    n_ok = 0
    cyc_sum = 0
    t0 = time.perf_counter()
    try:
        for i in range(args.n):
            ser.write(bytes([MAGIC_TX]) + pack_token(rng))
            mag = ser.read(1)
            if mag != bytes([MAGIC_RX]):
                raise SystemExit(f"token {i}: bad magic {mag!r}")
            rsp = ser.read(RSP)
            if len(rsp) != RSP:
                raise SystemExit(f"token {i}: short reply {len(rsp)}")
            cyc = int.from_bytes(rsp[192:196], "little")
            cyc_sum += cyc
            n_ok += 1
    finally:
        ser.close()
    wall = time.perf_counter() - t0
    tok_s = n_ok / wall if wall else 0.0
    avg_cyc = cyc_sum / n_ok if n_ok else 0.0
    cyc_tok_s = args.clk_hz / avg_cyc if avg_cyc else 0.0
    print(f"{n_ok} tokens / {wall:.4f} wall seconds")
    print(f"tok/s      {tok_s:.1f}   (wall clock)")
    print(f"avg_cycles {avg_cyc:.1f}")
    print(f"cyc tok/s  {cyc_tok_s:.1f}   (clk_hz / avg_cycles, {args.clk_hz:.0e} Hz)")


if __name__ == "__main__":
    main()
