"""Shared 4-bit W for xbar DUTs. N=8 matches ffn_tile_8x8_b4_reg; larger N is local quant."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BITS = 4
N = 8
REG_RTL = ROOT / "rtl" / "ffn_tile_8x8_b4_reg.v"
NPY = ROOT / "artifacts" / "tile8_int4_xbar.npy"


def _twos_nibble(v: int, bits: int = BITS) -> int:
    mask = (1 << bits) - 1
    raw = int(v) & mask
    sign = 1 << (bits - 1)
    return raw - (1 << bits) if raw & sign else raw


def parse_w_from_rtl(path: Path = REG_RTL, n: int = N, bits: int = BITS) -> np.ndarray:
    text = path.read_text()
    found = dict(re.findall(r"W_ROW(\d+)\s*=\s*\d+'h([0-9a-fA-F]+)", text))
    if len(found) < n:
        raise SystemExit(f"expected {n} W_ROWn in {path}")
    w = np.zeros((n, n), dtype=np.int8)
    for r in range(n):
        h = int(found[str(r)], 16)
        for c in range(n):
            w[r, c] = _twos_nibble(h >> (c * bits), bits)
    return w


def load_w8() -> np.ndarray:
    if NPY.exists():
        w = np.load(NPY)
        if w.shape == (N, N):
            return w.astype(np.int8)
    w = parse_w_from_rtl()
    NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(NPY, w)
    return w


def load_w(n: int, bits: int = BITS) -> np.ndarray:
    """N=8 keeps the signed-off xbar tile; N>8 is per-row quant of down_proj[:n,:n]."""
    if n == N and bits == BITS:
        return load_w8()
    local = ROOT / "artifacts" / f"tile{n}_local_int{bits}.npy"
    if local.exists():
        w = np.load(local)
        if tuple(w.shape) == (n, n):
            return w.astype(np.int8)
    from quant.load_ffn import load_down_proj
    from quant.quantize import quantize_per_row

    fp = load_down_proj(ROOT / "quant" / "cache")[:n, :n]
    q = quantize_per_row(fp, bits)
    local.parent.mkdir(parents=True, exist_ok=True)
    np.save(local, q.w_int)
    return q.w_int.astype(np.int8)


def bit_serial_matvec(w: np.ndarray, x: np.ndarray, in_w: int = 8) -> np.ndarray:
    """Two's-complement bit-serial: y += (W @ xbit)<<k, subtract on the sign bit."""
    acc = np.zeros(w.shape[0], dtype=np.int64)
    x = x.astype(np.int32)
    w32 = w.astype(np.int32)
    for k in range(in_w):
        xb = (x >> k) & 1
        partial = w32 @ xb
        if k == in_w - 1:
            acc -= partial << k
        else:
            acc += partial << k
    return acc.astype(np.int32)
