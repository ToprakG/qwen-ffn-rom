"""Integer golden for RMSNorm, attention decode, and one decoder layer.

Must match rtl/inv_rsqrt.v, rtl/rmsnorm.v, rtl/attn_decode.v, rtl/decoder_layer.v.
Old restoring isqrt/div remain as reference; RMSNorm uses Newton rsqrt.
"""

from __future__ import annotations

import numpy as np

from quant.delta_int import SHIFT, gated_delta_step, sat_sw

SAT8_MIN, SAT8_MAX = -128, 127


def restoring_isqrt(n: int) -> int:
    """Floor sqrt of a 32-bit unsigned int. Same 16-step restore as rtl/isqrt32.v."""
    n = int(n) & 0xFFFFFFFF
    root = 0
    rem = 0
    x = n
    for _ in range(16):
        rem = ((rem << 2) | ((x >> 30) & 3)) & 0xFFFFFFFF
        x = (x << 2) & 0xFFFFFFFF
        trial = ((root << 2) | 1) & 0xFFFFFFFF
        root = (root << 1) & 0xFFFF
        if rem >= trial:
            rem = (rem - trial) & 0xFFFFFFFF
            root = (root | 1) & 0xFFFF
    return int(root)


def restoring_div_u32(num: int, den: int) -> int:
    """Unsigned 32/16 restoring divide, 32 steps, 16-bit quotient. Matches rtl/idiv_u32.v."""
    num = int(num) & 0xFFFFFFFF
    den = max(int(den) & 0xFFFF, 1)
    q = 0
    r = 0
    for _ in range(32):
        r = ((r << 1) | ((num >> 31) & 1)) & 0xFFFFFFFF
        num = (num << 1) & 0xFFFFFFFF
        q = (q << 1) & 0xFFFFFFFF
        if r >= den:
            r -= den
            q = (q | 1) & 0xFFFFFFFF
    return int(q)


def sat8(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.int64), SAT8_MIN, SAT8_MAX).astype(np.int64)


def rmsnorm8(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y_i = sat8( (x_i * w_i * inv) >> 16 ) with inv = (1<<16) / max(1, isqrt(sum x^2))."""
    return rmsnorm_h(x, w)


def rmsnorm_h(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Width-H RMSNorm. H=8 matches rtl/rmsnorm8.v; H=16 matches rtl/rmsnorm.v."""
    from quant.rsqrt_int import rmsnorm_nr

    return rmsnorm_nr(x, w)


def attn_decode_int(q: np.ndarray, K: np.ndarray, V: np.ndarray, shift: int = SHIFT) -> np.ndarray:
    """One-head decode. K,V are (S, D). D-wide inner product per cache row.

    scores = (K @ q) >> shift
    o      = (V.T @ scores) >> shift
    Softmax is not in the RTL; this is the MAC skeleton only.
    """
    q = np.asarray(q, dtype=np.int64).reshape(-1)
    K = np.asarray(K, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    s, d = K.shape
    assert q.shape == (d,) and V.shape == (s, d)
    scores = (K @ q) >> shift
    o = (V.T @ scores) >> shift
    return o.astype(np.int64)


def decoder_layer_int(
    x: np.ndarray,
    S: np.ndarray,
    g: int,
    beta: int,
    w_n1: np.ndarray,
    w_n2: np.ndarray,
    w_ffn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One DeltaNet decoder layer, hidden=8, mixer D=4 on h[0:4].

    rms1 → mixer → residual → rms2 → 8x8 FFN tap → residual.
    """
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    assert x.shape == (8,)
    h = rmsnorm8(x, w_n1)
    q = k = v = h[:4]
    S, o = gated_delta_step(S, q, k, v, g, beta)
    mid = x.copy()
    mid[:4] = sat8(x[:4] + (o >> SHIFT))
    h2 = rmsnorm8(mid, w_n2)
    y = (np.asarray(w_ffn, dtype=np.int64) @ h2).astype(np.int64)
    out = sat8(mid + (y >> 7))
    return S, out


def qwen_layer_int(
    x: np.ndarray,
    mix_S: list[np.ndarray],
    kv: list[tuple[np.ndarray, np.ndarray]],
    use_attn: bool,
    g: int,
    beta: int,
    w_n1: np.ndarray,
    w_n2: np.ndarray,
    w_ffn: np.ndarray,
    d: int = 4,
    heads: int = 4,
    s_max: int = 8,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Complete layer at H=heads*d. mix_S is per-head (D,D). kv is per-head (K,V) with rows ≤ s_max."""
    h_dim = heads * d
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    assert x.shape == (h_dim,)
    h = rmsnorm_h(x, w_n1)
    mid = x.copy()
    new_kv = []
    new_S = []
    for hd in range(heads):
        sl = slice(hd * d, (hd + 1) * d)
        q = h[sl]
        if use_attn:
            K, V = kv[hd]
            row = q.reshape(1, -1)
            K = row if K.shape[0] == 0 else np.vstack([K, row])
            V = row if V.shape[0] == 0 else np.vstack([V, row])
            if K.shape[0] > s_max:
                K, V = K[-s_max:], V[-s_max:]
            o = attn_decode_int(q, K, V)
            new_S.append(mix_S[hd])
            new_kv.append((K, V))
        else:
            S_h, o = gated_delta_step(mix_S[hd], q, q, q, g, beta)
            new_S.append(S_h)
            new_kv.append(kv[hd])
        mid[sl] = sat8(x[sl] + (o >> SHIFT))
    h2 = rmsnorm_h(mid, w_n2)
    y = (np.asarray(w_ffn, dtype=np.int64) @ h2).astype(np.int64)
    out = sat8(mid + (y >> 7))
    return new_S, new_kv, out
