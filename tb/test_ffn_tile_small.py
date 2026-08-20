"""Bit-exact cocotb vs numpy for a small generated tile (PnR sizes)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import cocotb
from cocotb.triggers import Timer

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("FFN_TILE_N", "8"))
W_W = int(os.environ.get("FFN_TILE_BITS", "4"))
IN_W = 8
ACC_W = IN_W + W_W + int(math.ceil(math.log2(max(N, 2))))
W_PATH = ROOT / "artifacts" / f"tile{N}_int{W_W}.npy"


def pack_signed(xs: np.ndarray, width: int) -> int:
    mask = (1 << width) - 1
    v = 0
    for i, x in enumerate(xs.tolist()):
        v |= (int(x) & mask) << (i * width)
    return v


def unpack_signed(val: int, n: int, width: int) -> np.ndarray:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        raw = (val >> (i * width)) & mask
        out[i] = raw - (1 << width) if raw & sign else raw
    return out


@cocotb.test()
async def test_small_tile_bit_exact(dut):
    w = np.load(W_PATH).astype(np.int32)
    assert w.shape == (N, N), w.shape
    rng = np.random.default_rng(2)
    vecs = [
        np.zeros(N, dtype=np.int32),
        np.ones(N, dtype=np.int32),
        np.full(N, 127, dtype=np.int32),
        np.full(N, -128, dtype=np.int32),
    ]
    for i in range(N):
        e = np.zeros(N, dtype=np.int32)
        e[i] = 1
        vecs.append(e)
    vecs.append(rng.integers(-128, 128, size=N, dtype=np.int32))
    for x in vecs:
        dut.x_flat.value = pack_signed(x, IN_W)
        await Timer(1, unit="ns")
        y = unpack_signed(int(dut.y_flat.value), N, ACC_W)
        y_ref = w @ x
        np.testing.assert_array_equal(y, y_ref)
