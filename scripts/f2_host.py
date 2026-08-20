#!/usr/bin/env python3
"""Host for f2/cl_qwen_farm_mmio.v (mixer farm, not 35B-A3B).

  python scripts/f2_host.py --dry-run
  python scripts/f2_host.py --bar ocl   # on an F2 instance after AFI load

Without AWS FPGA SDK this only dry-runs. See f2/README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQ = 224
CTRL, STATUS, CYCLES = 0x000, 0x004, 0x008
REQ_BASE, RSP_BASE = 0x040, 0x100


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--clk-hz", type=float, default=250e6, help="CL clk_main_a0")
    p.add_argument("--bar", default="ocl")
    args = p.parse_args()

    if args.dry_run:
        # Same farm as UART top: 24 layers D=4, ~30 clk/layer × 16 heads lockstep
        cyc = 720
        print("F2 trial DUT = rtl/qwen08b_heads16_d4 (not 35B-A3B)")
        print(f"assumed {cyc} clk/token  @ {args.clk_hz/1e6:.0f} MHz → {args.clk_hz/cyc:.0f} tok/s")
        print("full 35B Q4 does not fit 16 GiB HBM; ASIC is 5k–23k @ 4k")
        print("dry-run: no PCI BAR")
        return

    try:
        import fpga_pci  # type: ignore  # AWS FPGA SDK on the instance
    except ImportError:
        print("fpga_pci not found. Run on the F2 AMI after HDK/SDK setup.", file=sys.stderr)
        sys.exit(1)

    handle = fpga_pci.fpga_pci_attach(0, 0, 0 if args.bar == "ocl" else 4, 0)
    rng_req = bytes(REQ)  # zeros: legal for a smoke kick
    t0 = time.perf_counter()
    for i in range(0, REQ, 4):
        w = int.from_bytes(rng_req[i : i + 4].ljust(4, b"\x00"), "little")
        fpga_pci.fpga_pci_poke(handle, REQ_BASE + i, w)
    ok = 0
    for _ in range(args.n):
        fpga_pci.fpga_pci_poke(handle, CTRL, 1)
        for _ in range(1_000_000):
            st = fpga_pci.fpga_pci_peek(handle, STATUS)
            if st & 2:
                ok += 1
                fpga_pci.fpga_pci_poke(handle, CTRL, 4)
                break
        else:
            print("timeout waiting done")
            break
    dt = time.perf_counter() - t0
    print(f"kicks {ok}/{args.n}  wall {ok/dt:.1f} tok/s  dt={dt:.3f}s")
    cyc = fpga_pci.fpga_pci_peek(handle, CYCLES)
    print(f"last CYCLES {cyc}  implied {args.clk_hz/max(cyc,1):.0f} tok/s")
    fpga_pci.fpga_pci_detach(handle)


if __name__ == "__main__":
    main()
