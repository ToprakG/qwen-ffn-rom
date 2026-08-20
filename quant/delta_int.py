"""Integer Gated DeltaNet recurrence — bit-exact golden for the RTL PE.

Qwen3.5 per-head float update (Yang et al. / transformers fallback):

    S ← exp(g) · S
    kv_mem = kᵀ S
    delta  = β · (v − kv_mem)
    S ← S + k ⊗ delta
    o  = qᵀ S

This module is the same dataflow with Q0.8 gates and arithmetic right-shifts
instead of float. L2-norm / √d_k live outside the PE (same split as the FFN
tile: RTL is integer, scale is software).

S is (D, D) with axis 0 = key dim, axis 1 = value dim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHIFT = 8
S_W = 16
O_W = 24
QK_W = 8
V_W = 8
G_W = 8
S_MIN = -(1 << (S_W - 1))
S_MAX = (1 << (S_W - 1)) - 1


def sat_sw(x: np.ndarray | int) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), S_MIN, S_MAX).astype(np.int64)


def asr(x: np.ndarray | int, shift: int = SHIFT) -> np.ndarray:
    return np.asarray(x, dtype=np.int64) >> shift


@dataclass
class DeltaIntState:
    S: np.ndarray  # (D, D) int64

    @classmethod
    def zeros(cls, d: int) -> "DeltaIntState":
        return cls(S=np.zeros((d, d), dtype=np.int64))


def gated_delta_step(
    S: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: int,
    beta: int,
    shift: int = SHIFT,
) -> tuple[np.ndarray, np.ndarray]:
    """One token. Returns (S_next, o) as int64."""
    S = np.asarray(S, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64).reshape(-1)
    k = np.asarray(k, dtype=np.int64).reshape(-1)
    v = np.asarray(v, dtype=np.int64).reshape(-1)
    d = S.shape[0]
    assert S.shape == (d, d)
    assert q.shape == k.shape == v.shape == (d,)

    s_decay = asr(S * int(g), shift)
    kv = asr(s_decay.T @ k, shift)
    delta = asr(int(beta) * (v - kv), shift)
    outer = asr(np.outer(k, delta), shift)
    s_next = sat_sw(s_decay + outer)
    o = asr(s_next.T @ q, shift)
    return s_next, o


def run_sequence(
    tokens: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]],
    d: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """tokens: list of (q, k, v, g, beta). Returns (outputs, final S)."""
    S = np.zeros((d, d), dtype=np.int64)
    outs: list[np.ndarray] = []
    for q, k, v, g, beta in tokens:
        S, o = gated_delta_step(S, q, k, v, g, beta)
        outs.append(o)
    return outs, S
