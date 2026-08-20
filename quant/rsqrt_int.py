"""Integer Newton-Raphson inverse-sqrt. Bit-exact vs rtl/inv_rsqrt.v.

inv ≈ (1<<16) / sqrt(ssq), same Q16 as the old restoring isqrt+div path.
One Newton iteration after a 64-entry LUT seed. Not bit-exact to restoring
floor(sqrt) then divide — bounded: int8 RMS differs by at most 1 vs restoring
at H=8/16 (see scripts/rsqrt_quality.py).
"""

from __future__ import annotations

import math

import numpy as np

SAT8_MIN, SAT8_MAX = -128, 127

LUT_BITS = 6
SEED_W = 19
INV_W = 17


def rsqrt_lut() -> list[int]:
    tab = []
    for i in range(1 << LUT_BITS):
        lo = i << (32 - LUT_BITS)
        hi = (i + 1) << (32 - LUT_BITS)
        mid = (lo + hi) // 2
        tab.append(0 if mid == 0 else int(round((1 << 31) / math.sqrt(mid))))
    return tab


RSQRT_LUT = rsqrt_lut()


def clz32(x: int) -> int:
    """Match rtl/inv_rsqrt.v: binary nibble climb, 32 if x==0."""
    x = int(x) & 0xFFFFFFFF
    if x == 0:
        return 32
    n = 0
    if (x & 0xFFFF0000) == 0:
        n += 16
        x <<= 16
    if (x & 0xFF000000) == 0:
        n += 8
        x <<= 8
    if (x & 0xF0000000) == 0:
        n += 4
        x <<= 4
    if (x & 0xC0000000) == 0:
        n += 2
        x <<= 2
    if (x & 0x80000000) == 0:
        n += 1
    return n


def inv_rsqrt_q16(ssq: int) -> int:
    """Q16 inv = (1<<16)/sqrt(ssq). Combo, one Newton step. Matches rtl/inv_rsqrt.v."""
    x = max(int(ssq), 1) & 0xFFFFFFFF
    lz = clz32(x)
    sh = lz & ~1
    xn = (x << sh) & 0xFFFFFFFF
    idx = xn >> (32 - LUT_BITS)
    seed = RSQRT_LUT[idx]
    y2 = seed * seed
    xyy_q16 = (xn * y2) >> 46
    inner = (3 << 16) - xyy_q16
    if inner < 0:
        inner = 0
    y = (seed * inner) >> 17
    rsh = 15 - (sh >> 1)
    inv = y >> rsh
    if inv <= 0:
        inv = 1
    cap = (1 << INV_W) - 1
    if inv > cap:
        inv = cap
    return int(inv)


def sat8_arr(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), SAT8_MIN, SAT8_MAX).astype(np.int64)


def rmsnorm_inv(x: np.ndarray, w: np.ndarray, inv: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    w = np.asarray(w, dtype=np.int64).reshape(-1)
    y = (x * w * int(inv)) >> 16
    return sat8_arr(y)


def rmsnorm_nr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fast RMSNorm. Matches rtl/rmsnorm.v after the NR unit."""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    ssq = int((x * x).sum())
    return rmsnorm_inv(x, w, inv_rsqrt_q16(ssq))
