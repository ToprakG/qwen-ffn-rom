"""Integer online-softmax attention (FlashAttention-2 decode block).

Bit-exact vs rtl/attn_online.v. One pass over KV, no S-length score vector.
int4 KV is the cache format; q stays int8.
"""

from __future__ import annotations

import math

import numpy as np

SHIFT = 8
EXP_Q = 8
EXP_MAX = 32
O_W = 24
KV_W = 4
QK_W = 8

# 256 * exp(-k/8), k = m - s in integer score units. Same ROM as rtl/attn_exp_lut.v.
EXP_TAB = [
    max(0, int(round((1 << EXP_Q) * math.exp(-k / 8.0)))) for k in range(EXP_MAX + 1)
]


def i32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def asr(x: int, n: int) -> int:
    return i32(i32(x) >> n)


def tdiv(n: int, d: int) -> int:
    """Toward-zero divide, matches Verilog signed / in Verilator."""
    d = max(abs(int(d)), 1)
    n = int(n)
    if n < 0:
        return -((-n) // d)
    return n // d


def sat_ow(x: int, bits: int = O_W) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return int(np.clip(x, lo, hi))


def exp_lut(delta: int) -> int:
    d = min(max(int(delta), 0), EXP_MAX)
    return EXP_TAB[d]


def quant_int4(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int4 per D-channel. Standard KV-cache quant."""
    x = np.asarray(x, dtype=np.float64)
    amax = np.maximum(np.max(np.abs(x), axis=0, keepdims=True), 1e-8)
    scale = amax / 7.0
    q = np.clip(np.round(x / scale), -8, 7).astype(np.int64)
    return q, scale


def pack_int4_asr(x: np.ndarray) -> np.ndarray:
    """int8 → int4 by arithmetic shift 4 (scale 16). RTL cache stores these codes."""
    x = np.asarray(x, dtype=np.int64)
    return np.clip(x >> 4, -8, 7).astype(np.int64)


def attn_fp32(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = q.shape[0]
    logits = (K @ q) / math.sqrt(max(d, 1))
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / max(p.sum(), 1e-12)
    return p @ V


def attn_online_int(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    p_lanes: int,
    shift: int = SHIFT,
) -> np.ndarray:
    """One head. K,V are (S, D) int (int4 codes or int8). q is int8 (D,).

    Block-online softmax over groups of p_lanes positions. No score vector of
    length S is materialized.
    """
    q = np.asarray(q, dtype=np.int64).reshape(-1)
    K = np.asarray(K, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    s_len, d = K.shape
    assert q.shape == (d,) and V.shape == (s_len, d)
    p = int(p_lanes)
    n_steps = (s_len + p - 1) // p

    m = 0
    have = False
    ell = 0
    o = np.zeros(d, dtype=np.int64)

    for step in range(n_steps):
        scores = []
        valid = []
        for lane in range(p):
            t = step * p + lane
            if t < s_len:
                sc = i32(int((K[t] * q).sum()) >> shift)
                scores.append(sc)
                valid.append(True)
            else:
                scores.append(0)
                valid.append(False)
        m_blk = max(sc for sc, ok in zip(scores, valid) if ok)
        if not have:
            m_new = m_blk
            scale = 0
            have = True
        else:
            m_new = m_blk if m_blk > m else m
            scale = exp_lut(m_new - m)
        ws = []
        acc_w = 0
        for sc, ok in zip(scores, valid):
            w = exp_lut(m_new - sc) if ok else 0
            ws.append(w)
            acc_w += w
        ell = asr(ell * scale, EXP_Q) + acc_w
        o_s = np.array([asr(int(o[j]) * scale, EXP_Q) for j in range(d)], dtype=np.int64)
        for lane in range(p):
            t = step * p + lane
            if t >= s_len:
                continue
            w = ws[lane]
            for j in range(d):
                o_s[j] = i32(int(o_s[j]) + int(w) * int(V[t, j]))
        o = o_s
        m = m_new

    inv_scale = 1 << EXP_Q
    out = np.empty(d, dtype=np.int64)
    den = max(ell, 1)
    for j in range(d):
        out[j] = sat_ow(tdiv(int(o[j]) * inv_scale, den))
    return out


def attn_online_gqa_int(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    p_lanes: int,
    n_q_per_kv: int,
    shift: int = SHIFT,
) -> np.ndarray:
    """Q is (n_kv * n_q_per, D). K,V are (n_kv, S, D). Reuse each KV across n_q_per Q."""
    Q = np.asarray(Q, dtype=np.int64)
    K = np.asarray(K, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    n_kv, s_len, d = K.shape
    n_q = n_kv * n_q_per_kv
    assert Q.shape == (n_q, d) and V.shape == (n_kv, s_len, d)
    outs = []
    for kv in range(n_kv):
        for qi in range(n_q_per_kv):
            h = kv * n_q_per_kv + qi
            outs.append(attn_online_int(Q[h], K[kv], V[kv], p_lanes, shift=shift))
    return np.stack(outs, axis=0)
