#!/usr/bin/env python3
"""Generate CSD/shift-add Verilog for a quantized FFN tile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtl_gen.csd import eval_csd, signed_range
from rtl_gen.emit_tile import write_tile


def csd_matvec(w_int: np.ndarray, x_int: np.ndarray) -> np.ndarray:
    """Numpy CSD mat-vec: the equivalence baseline the RTL must match."""
    rows, cols = w_int.shape
    y = np.zeros(rows, dtype=np.int64)
    x_int = x_int.astype(np.int64)
    for i in range(rows):
        acc = 0
        for j in range(cols):
            acc += eval_csd(int(w_int[i, j]), int(x_int[j]))
        y[i] = acc
    return y.astype(np.int32)


def check_csd(bits: int) -> None:
    for w in signed_range(bits):
        for x in (-128, -7, -1, 0, 1, 7, 127):
            got = eval_csd(w, x)
            exp = w * x
            if got != exp:
                raise SystemExit(f"CSD mismatch w={w} x={x}: {got} != {exp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--in-w", type=int, default=8)
    parser.add_argument("--registered", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    check_csd(args.bits)
    w_path = ROOT / "artifacts" / f"tile{args.tile}_int{args.bits}.npy"
    if not w_path.exists():
        raise SystemExit(f"missing {w_path}; run scripts/quantize_ffn.py first")
    w = np.load(w_path)
    x_path = ROOT / "artifacts" / "ref_x_int8.npy"
    y_path = ROOT / "artifacts" / "ref_y_int.npy"
    if w.shape[0] == args.tile and x_path.exists() and args.tile == w.shape[1]:
        x = np.load(x_path).astype(np.int32)
        if x.size != args.tile:
            x = x[: args.tile]
        y_dot = w.astype(np.int32) @ x
        y_csd = csd_matvec(w, x)
        if not np.array_equal(y_dot, y_csd):
            raise SystemExit("CSD mat-vec != integer W@x")
        if args.tile == 128 and y_path.exists():
            y_ref = np.load(y_path)
            if not np.array_equal(y_csd, y_ref):
                raise SystemExit("CSD mat-vec != artifacts/ref_y_int.npy")
        print("numpy CSD mat-vec matches W_int @ x_int")
    if args.out is None:
        if args.tile == 128 and args.bits == 4 and not args.registered:
            out = ROOT / "rtl" / f"ffn_tile_{args.tile}x{args.tile}.v"
        else:
            suffix = "_reg" if args.registered else ""
            out = ROOT / "rtl" / f"ffn_tile_{args.tile}x{args.tile}_b{args.bits}{suffix}.v"
    else:
        out = args.out
    write_tile(
        out,
        w,
        args.bits,
        in_w=args.in_w,
        module="ffn_tile",
        registered=args.registered,
        reg_module="ffn_tile_reg",
    )
    nz = int(np.count_nonzero(w))
    print(f"wrote {out}  shape={w.shape}  bits={args.bits}  nonzero={nz}/{w.size}")


if __name__ == "__main__":
    main()
