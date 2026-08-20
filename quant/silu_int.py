"""Integer SiLU (Q3) and SwiGLU output-stage fold. Bit-exact vs rtl/silu_lut.v.

silu(x) = sat8(round(x * sigmoid(x/8))). Applied on int8 after a tap >>7.
SwiGLU lane: sat8((silu(g) * u) >> 7). Combo on the farm output register —
zero extra handshake cycles vs ffn_tap_unit (still 2 clk).
"""

from __future__ import annotations

import math

import numpy as np

from quant.layer_int import SAT8_MAX, SAT8_MIN


def silu_q3_scalar(x: int) -> int:
    x = int(np.clip(int(x), SAT8_MIN, SAT8_MAX))
    y = int(round(x * (1.0 / (1.0 + math.exp(-x / 8.0)))))
    return int(np.clip(y, SAT8_MIN, SAT8_MAX))


SILU_LUT = [silu_q3_scalar(i) for i in range(-128, 128)]


def silu_q3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    return np.array([SILU_LUT[int(v) + 128] for v in x], dtype=np.int64)


def sat8_arr(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), SAT8_MIN, SAT8_MAX).astype(np.int64)


def swiglu_out(gate_acc: np.ndarray, up: np.ndarray, tap_shift: int = 7) -> np.ndarray:
    """Farm output stage: sat8(tap>>shift) → SiLU → ×up → sat8(>>7)."""
    g8 = sat8_arr(np.asarray(gate_acc, dtype=np.int64).reshape(-1) >> tap_shift)
    u8 = sat8_arr(np.asarray(up, dtype=np.int64).reshape(-1))
    s = silu_q3(g8)
    return sat8_arr((s * u8) >> 7)
