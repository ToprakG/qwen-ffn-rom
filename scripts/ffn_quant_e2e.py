#!/usr/bin/env python3
"""Same layer-0 SwiGLU e2e metric as ffn_structure_probe.py, for Q2/Q3/Q4 + codebooks.

  python scripts/ffn_quant_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ffn_structure_probe import (  # noqa: E402
    CACHE,
    CELL,
    EFF,
    BITS as ROM_Q4_BITS,
    RETICLE,
    HIDDEN,
    INTER,
    N_LAYERS,
    N_X,
    mlp_name,
    rom_mm2,
    silu,
    swiglu_down,
    unit_rms_x,
)

N_FIT = 200_000


def load_layer0() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = {p: CACHE / (mlp_name(0, p).replace(".", "__") + ".f16.npy") for p in ("gate", "up", "down")}
    return tuple(np.load(names[p]).astype(np.float32) for p in ("gate", "up", "down"))  # type: ignore[return-value]


def rel_f(w: np.ndarray, w_hat: np.ndarray) -> float:
    return float(np.linalg.norm(w - w_hat) / (np.linalg.norm(w) + 1e-12))


def rel_act(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    y, y_hat = w @ x, w_hat @ x
    r = np.linalg.norm(y - y_hat, axis=0) / (np.linalg.norm(y, axis=0) + 1e-12)
    return float(r.mean())


def e2e(wg, wu, wd, hats, x, y_true) -> float:
    y_h = swiglu_down(hats["gate"], hats["up"], hats["down"], x)
    r = np.linalg.norm(y_true - y_h, axis=0) / (np.linalg.norm(y_true, axis=0) + 1e-12)
    return float(r.mean())


def area_bits(bits: float) -> dict:
    n = 3 * HIDDEN * INTER * N_LAYERS
    mm2 = rom_mm2(n) * (bits / ROM_Q4_BITS)
    return {
        "ffn_params_stored_as": f"{bits}-bit",
        "rom_mm2": round(mm2, 1),
        "pct_reticle": round(100.0 * mm2 / RETICLE, 1),
    }


def quant_row_sym(w: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    amax = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.maximum(amax / max(qmax, 1), 1e-12)
    q = np.clip(np.rint(w / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)


def quant_group_asymm(w: np.ndarray, bits: int, gs: int) -> np.ndarray:
    qmax = (1 << bits) - 1
    out, inn = w.shape
    pad = (gs - inn % gs) % gs
    g = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = g.reshape(out, -1, gs)
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    scale = np.maximum((hi - lo) / qmax, 1e-12)
    q = np.clip(np.rint((g - lo) / scale), 0, qmax)
    recon = q * scale + lo
    return recon.reshape(out, -1)[:, :inn].astype(np.float32)


def lloyd_1d_group(w: np.ndarray, k: int, gs: int, iters: int = 6) -> np.ndarray:
    """Per-group 1D k-means, K centroids = a tiny codebook (K=4 → 2-bit index)."""
    out, inn = w.shape
    pad = (gs - inn % gs) % gs
    g = np.pad(w, ((0, 0), (0, pad))) if pad else w.copy()
    g = g.reshape(out, -1, gs)
    n_g = g.shape[1]
    # init: linspace min..max
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    cb = lo + (hi - lo) * np.linspace(0, 1, k, dtype=np.float32).reshape(1, 1, k)
    for _ in range(iters):
        # (out, ng, gs, 1) vs (out, ng, 1, k)
        d = (g[..., None] - cb[..., None, :]) ** 2
        idx = d.argmin(axis=-1)
        for j in range(k):
            mask = idx == j
            cnt = mask.sum(axis=-1, keepdims=True).astype(np.float32)
            tot = (g * mask).sum(axis=-1, keepdims=True)
            upd = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
            cb[..., j] = np.where(cnt[..., 0] > 0, upd[..., 0], cb[..., j])
    d = (g[..., None] - cb[..., None, :]) ** 2
    idx = d.argmin(axis=-1)
    recon = np.take_along_axis(cb, idx, axis=-1)
    return recon.reshape(out, -1)[:, :inn].astype(np.float32)


def _assign(x: np.ndarray, cb: np.ndarray, chunk: int = 4096) -> np.ndarray:
    cb_sq = (cb * cb).sum(axis=1)
    idx = np.empty(x.shape[0], dtype=np.int32)
    for i0 in range(0, x.shape[0], chunk):
        sl = x[i0 : i0 + chunk]
        dots = sl @ cb.T
        idx[i0 : i0 + chunk] = np.argmin(cb_sq[None, :] - 2.0 * dots, axis=1)
    return idx


def _kmeans(x: np.ndarray, k: int, iters: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    pick = rng.choice(n, size=k, replace=False)
    cb = x[pick].copy()
    for _ in range(iters):
        idx = _assign(x, cb)
        for j in range(k):
            m = idx == j
            if np.any(m):
                cb[j] = x[m].mean(axis=0)
    return cb


def vq_residual(w: np.ndarray, dim: int, k: int, stages: int, seed: int) -> np.ndarray:
    """Residual VQ on last-axis groups. bits/weight ≈ stages * log2(k) / dim."""
    out, inn = w.shape
    pad = (dim - inn % dim) % dim
    g = np.pad(w, ((0, 0), (0, pad))) if pad else w
    vecs = np.ascontiguousarray(g.reshape(-1, dim))
    rng = np.random.default_rng(seed)
    fit_n = min(N_FIT, vecs.shape[0])
    fit = vecs[rng.choice(vecs.shape[0], size=fit_n, replace=False)]
    residual = fit.copy()
    cbs = []
    for s in range(stages):
        cb = _kmeans(residual, k, iters=7, seed=seed + s)
        cbs.append(cb)
        residual = residual - cb[_assign(residual, cb)]
    recon = np.zeros_like(vecs)
    rest = vecs
    for cb in cbs:
        idx = _assign(rest, cb)
        recon += cb[idx]
        rest = rest - cb[idx]
    out_w = recon.reshape(out, -1)[:, :inn]
    return out_w.astype(np.float32)


def apply_all(fn, wg, wu, wd):
    return {"gate": fn(wg), "up": fn(wu), "down": fn(wd)}


def score(name: str, bits: float, hats, wg, wu, wd, x, y_true, h_true) -> dict:
    rec = {
        "bits": bits,
        "area": area_bits(bits),
        "rel_f": {p: rel_f(w, hats[p]) for p, w in (("gate", wg), ("up", wu), ("down", wd))},
        "rel_act": {
            "gate": rel_act(wg, hats["gate"], x),
            "up": rel_act(wu, hats["up"], x),
            "down_on_true_h": rel_act(wd, hats["down"], h_true),
        },
        "e2e_swiglu": e2e(wg, wu, wd, hats, x, y_true),
    }
    print(
        f"  {name:36s}  e2e={rec['e2e_swiglu']:.4f}  "
        f"gateF={rec['rel_f']['gate']:.4f}  downAct_h={rec['rel_act']['down_on_true_h']:.4f}  "
        f"{rec['area']['rom_mm2']:.0f} mm²",
        flush=True,
    )
    return rec


def main() -> None:
    t0 = time.time()
    print("load layer 0 ...", flush=True)
    wg, wu, wd = load_layer0()
    x = unit_rms_x(HIDDEN, N_X, seed=1)
    y_true = swiglu_down(wg, wu, wd, x)
    h_true = silu(wg @ x) * (wu @ x)

    methods = {}

    print("Q4 / Q3 / Q2 row-wise ...", flush=True)
    for bits in (4, 3, 2):
        hats = apply_all(lambda w, b=bits: quant_row_sym(w, b), wg, wu, wd)
        methods[f"row_sym_q{bits}"] = score(f"row_sym_q{bits}", bits, hats, wg, wu, wd, x, y_true, h_true)

    print("Q4 group64 minmax (fair 4-bit baseline) ...", flush=True)
    hats = apply_all(lambda w: quant_group_asymm(w, 4, 64), wg, wu, wd)
    methods["group64_minmax_q4"] = score(
        "group64_minmax_q4", 4.0, hats, wg, wu, wd, x, y_true, h_true
    )

    print("Q2 groupwise minmax ...", flush=True)
    for gs in (32, 64, 128):
        hats = apply_all(lambda w, g=gs: quant_group_asymm(w, 2, g), wg, wu, wd)
        methods[f"group{gs}_minmax_q2"] = score(
            f"group{gs}_minmax_q2", 2.0, hats, wg, wu, wd, x, y_true, h_true
        )

    print("Q2 group64 1D codebook K=4 (Lloyd) ...", flush=True)
    hats = apply_all(lambda w: lloyd_1d_group(w, k=4, gs=64), wg, wu, wd)
    methods["group64_kmeans4_q2"] = score(
        "group64_kmeans4_q2", 2.0, hats, wg, wu, wd, x, y_true, h_true
    )

    print("VQ 4-dim K=256 (2 bit/weight) ...", flush=True)
    hats = {}
    for name, w in (("gate", wg), ("up", wu), ("down", wd)):
        print(f"  vq {name} ...", flush=True)
        hats[name] = vq_residual(w, dim=4, k=256, stages=1, seed=0)
    methods["vq4_k256_2bpp"] = score("vq4_k256_2bpp", 2.0, hats, wg, wu, wd, x, y_true, h_true)

    print("residual VQ 8-dim 2×K=256 (2 bit/weight, AQLM-shaped) ...", flush=True)
    hats = {}
    for name, w in (("gate", wg), ("up", wu), ("down", wd)):
        print(f"  rvq {name} ...", flush=True)
        hats[name] = vq_residual(w, dim=8, k=256, stages=2, seed=1)
    methods["rvq8_2x256_2bpp"] = score("rvq8_2x256_2bpp", 2.0, hats, wg, wu, wd, x, y_true, h_true)

    # Q3 groupwise — often the coding-sane compromise
    print("Q3 group64 minmax ...", flush=True)
    hats = apply_all(lambda w: quant_group_asymm(w, 3, 64), wg, wu, wd)
    methods["group64_minmax_q3"] = score(
        "group64_minmax_q3", 3.0, hats, wg, wu, wd, x, y_true, h_true
    )

    out = {
        "checkpoint": "Qwen/Qwen3.8-27B",
        "layer": 0,
        "x": "unit-RMS Gaussian n=128, same seed as ffn_structure_probe.py",
        "compare_to_structure_e2e": {
            "sparse_2_4": 0.422,
            "svd_indep_r1024": 0.543,
            "monarch_b4": 0.633,
            "monarch_b8": 0.864,
            "joint_gate_up_r1024_down_dense": 0.419,
        },
        "methods": methods,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    path = ROOT / "artifacts" / "ffn_quant_e2e.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} in {out['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
