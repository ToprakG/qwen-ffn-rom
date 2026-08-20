#!/usr/bin/env python3
"""Step 0: post-hoc structure probe on one Qwen3.8-27B FFN layer.

Downloads gate/up/down via HTTP range (not the full 55 GB checkpoint). Reports
SVD energy, 2:4 / Monarch / joint-gate-up reconstruction, activation error on
unit-RMS Gaussian x (post-RMSNorm model), and adjacent-layer cosine/CKA.

  python scripts/ffn_structure_probe.py
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CACHE = ROOT / "quant" / "cache" / "qwen38_27b"

REPO = "Qwen/Qwen3.8-27B"
HF_RESOLVE = f"https://huggingface.co/{REPO}/resolve/main"
USER_AGENT = "qwen-ffn-rom-structure-probe"

# Same 28 nm Q4 via-ROM estimator as scripts/draw_area_q4_28nm.py
CELL = 0.015875
EFF = 0.62
BITS = 4
RETICLE = 26.0 * 33.0
N_LAYERS = 64
HIDDEN = 5120
INTER = 17408

PROBE_LAYER = 0
SIM_LAYERS = (0, 1, 62, 63)
N_X = 128
RSVD_K = 1536
RANKS = (256, 512, 1024, 2048, 3072, 4096)
MONARCH_BLOCKS = (4, 8)


def rom_mm2(n_params: float) -> float:
    return n_params * BITS * CELL / EFF / 1e6


def hf_get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


def load_index() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "model.safetensors.index.json"
    if not path.exists():
        path.write_bytes(hf_get(f"{HF_RESOLVE}/model.safetensors.index.json"))
    return json.loads(path.read_text())


def safetensor_header(shard: str) -> dict:
    path = CACHE / f"{shard}.header.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"{HF_RESOLVE}/{shard}"
    hdr_len = struct.unpack("<Q", hf_get(url, 0, 7))[0]
    header = json.loads(hf_get(url, 8, 7 + hdr_len).decode("utf-8"))
    header["_hdr_len"] = hdr_len
    path.write_text(json.dumps(header))
    return header


def bf16_to_fp32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).copy()


def load_weight(name: str, weight_map: dict) -> np.ndarray:
    npy = CACHE / (name.replace(".", "__") + ".f16.npy")
    if npy.exists():
        return np.load(npy).astype(np.float32, copy=False)
    shard = weight_map[name]
    header = safetensor_header(shard)
    info = header[name]
    if info["dtype"] != "BF16":
        raise ValueError(f"{name} dtype {info['dtype']}")
    off0, off1 = info["data_offsets"]
    base = 8 + int(header["_hdr_len"])
    raw = hf_get(f"{HF_RESOLVE}/{shard}", base + off0, base + off1 - 1)
    if len(raw) != off1 - off0:
        raise RuntimeError(f"short read {name}: {len(raw)} != {off1 - off0}")
    w = bf16_to_fp32(raw).reshape(info["shape"])
    np.save(npy, w.astype(np.float16))
    return w


def mlp_name(layer: int, proj: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.{proj}_proj.weight"


def rsvd(a: np.ndarray, k: int, n_iter: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized thin SVD. Returns U, S, Vt with k components."""
    rng = np.random.default_rng(seed)
    m, n = a.shape
    k = min(k, m, n)
    p = min(k + 12, min(m, n))
    if m >= n:
        omega = rng.standard_normal((n, p), dtype=np.float32)
        y = a @ omega
        for _ in range(n_iter):
            y = a @ (a.T @ y)
        q, _ = np.linalg.qr(y, mode="reduced")
        b = q.T @ a
        u_hat, s, vt = np.linalg.svd(b, full_matrices=False)
        return (q @ u_hat)[:, :k], s[:k].astype(np.float64), vt[:k]
    u, s, vt = rsvd(a.T, k, n_iter=n_iter, seed=seed)
    return vt.T, s, u.T


def rel_f_from_s(s: np.ndarray, fro: float, r: int) -> float:
    captured = float(np.dot(s[:r], s[:r]))
    tail = max(fro * fro - captured, 0.0)
    return float(np.sqrt(tail) / fro)


def activation_err(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> dict:
    y = w @ x
    y_hat = w_hat @ x
    num = np.linalg.norm(y - y_hat, axis=0)
    den = np.linalg.norm(y, axis=0) + 1e-12
    ratios = num / den
    return {
        "mean": float(ratios.mean()),
        "p90": float(np.quantile(ratios, 0.90)),
        "max": float(ratios.max()),
    }


def activation_err_factors(w: np.ndarray, apply_hat, x: np.ndarray) -> dict:
    y = w @ x
    y_hat = apply_hat(x)
    num = np.linalg.norm(y - y_hat, axis=0)
    den = np.linalg.norm(y, axis=0) + 1e-12
    ratios = num / den
    return {
        "mean": float(ratios.mean()),
        "p90": float(np.quantile(ratios, 0.90)),
        "max": float(ratios.max()),
    }


def prune_2_4(w: np.ndarray) -> np.ndarray:
    """Keep 2 of every 4 consecutive input-side weights, by magnitude."""
    out, inn = w.shape
    pad = (4 - inn % 4) % 4
    g = np.pad(w, ((0, 0), (0, pad))) if pad else w
    grouped = np.array(g, copy=True).reshape(out, -1, 4)
    keep = np.argpartition(np.abs(grouped), -2, axis=-1)[..., -2:]
    mask = np.zeros_like(grouped, dtype=bool)
    ii = np.arange(out)[:, None, None]
    jj = np.arange(grouped.shape[1])[None, :, None]
    mask[ii, jj, keep] = True
    grouped *= mask
    out_w = grouped.reshape(out, -1)
    return out_w[:, :inn] if pad else out_w


def silu(z: np.ndarray) -> np.ndarray:
    return z * (1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))


def swiglu_down(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, x: np.ndarray) -> np.ndarray:
    return wd @ (silu(wg @ x) * (wu @ x))


def unit_rms_x(d: int, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((d, n), dtype=np.float32)
    rms = np.sqrt((x * x).mean(axis=0, keepdims=True) + 1e-12)
    return x / rms


def monarch_rank_for_nblocks(nblocks: int, out: int, inn: int) -> int:
    """Rank per 4D slice so param count matches HazyResearch MonarchLinear(nblocks)."""
    short = min(out, inn)
    return max(1, short // (nblocks * nblocks))


def monarch_params(nblocks: int, out: int, inn: int) -> int:
    r = monarch_rank_for_nblocks(nblocks, out, inn)
    return r * nblocks * (out + inn)


def monarch_project(m: np.ndarray, nblocks: int):
    """Analytical rectangular Monarch projection (rank-r SVD on the b×b grid).

    Matches fly `blockdiag_butterfly_project_einsum_rank` with nblocks1=nblocks2=b
    and rank = min(out,in)/b², i.e. the MonarchLinear(nblocks=b) parameter budget.
    """
    out, inn = m.shape
    k = j = nblocks
    i = inn // k
    ldim = out // j
    rank = monarch_rank_for_nblocks(nblocks, out, inn)
    # einops '(l j) (k i) -> k j l i'
    mp = m.reshape(ldim, j, k, i).transpose(2, 1, 0, 3)
    k_, j_, l_, i_ = mp.shape
    blocks = mp.reshape(k_ * j_, l_, i_)
    u_b = np.empty((k_ * j_, l_, rank), dtype=np.float32)
    vt_b = np.empty((k_ * j_, rank, i_), dtype=np.float32)
    tail_sq = 0.0
    for bi, blk in enumerate(blocks):
        u, s, vt = np.linalg.svd(blk, full_matrices=False)
        u_b[bi] = (u[:, :rank] * s[:rank]).astype(np.float32)
        vt_b[bi] = vt[:rank].astype(np.float32)
        tail_sq += float(np.dot(s[rank:], s[rank:]))
    u_b = u_b.reshape(k_, j_, l_, rank)
    vt_b = vt_b.reshape(k_, j_, rank, i_)
    # w1: (k, r, j, i) ; w2 stored as U: (k, j, l, r)
    w1 = vt_b.transpose(0, 2, 1, 3).copy()
    w2 = u_b
    rel_f = float(np.sqrt(tail_sq) / np.linalg.norm(m))
    return w1, w2, rel_f, rank


def monarch_apply(w1: np.ndarray, w2: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x: (in, batch) → (out, batch). Inverse of monarch_project layout."""
    k, r, j, i = w1.shape
    _, _, ldim, _ = w2.shape
    batch = x.shape[1]
    xr = x.T.reshape(batch, k, i)
    # einsum 'b k i, k r j i, k j l r -> b l j'
    t = np.einsum("bki,krji->bkrj", xr, w1)
    y = np.einsum("bkrj,kjlr->blj", t, w2)
    return y.reshape(batch, ldim * j).T


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA, rows are examples."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    return float(xy / (xx * yy + 1e-12))


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def svd_report(name: str, w: np.ndarray, x: np.ndarray) -> dict:
    fro = float(np.linalg.norm(w))
    t0 = time.time()
    u, s, vt = rsvd(w, RSVD_K)
    rec = {
        "shape": list(w.shape),
        "params": int(w.size),
        "fro": fro,
        "rsvd_k": int(s.size),
        "rsvd_s": [float(v) for v in s[:64]],
        "energy_at_rank": {},
        "rel_f_at_rank": {},
        "rel_act_at_rank": {},
        "rsvd_sec": round(time.time() - t0, 2),
    }
    total_e = fro * fro
    for r in RANKS:
        if r > s.size:
            continue
        e = float(np.dot(s[:r], s[:r]) / total_e)
        rec["energy_at_rank"][str(r)] = e
        rec["rel_f_at_rank"][str(r)] = rel_f_from_s(s, fro, r)
        w_hat = (u[:, :r] * s[:r].astype(np.float32)) @ vt[:r]
        rec["rel_act_at_rank"][str(r)] = activation_err(w, w_hat, x)
        del w_hat
    rec["energy_curve"] = [float(np.dot(s[:r], s[:r]) / total_e) for r in range(1, s.size + 1, 16)]
    rec["energy_curve_ranks"] = list(range(1, s.size + 1, 16))
    return rec, (u, s, vt)


def method_area(per_layer_params: float) -> dict:
    n = per_layer_params * N_LAYERS
    mm2 = rom_mm2(n)
    return {
        "ffn_params": int(n),
        "rom_mm2": round(mm2, 1),
        "pct_reticle": round(100.0 * mm2 / RETICLE, 1),
        "compression": round((3 * HIDDEN * INTER * N_LAYERS) / n, 2),
    }


def probe_layer(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, x_h: np.ndarray) -> dict:
    # x for down is the SwiGLU hidden, not hidden-state. Build real-ish h from dense.
    h_true = silu(wg @ x_h) * (wu @ x_h)
    # unit-RMS on hidden too, plus the true h (anisotropic)
    x_i = unit_rms_x(INTER, N_X, seed=2)

    out: dict = {"projections": {}, "methods": {}}

    for name, w, x in (
        ("gate", wg, x_h),
        ("up", wu, x_h),
        ("down", wd, h_true),
        ("down_isotropic", wd, x_i),
    ):
        print(f"  SVD {name} {tuple(w.shape)} ...", flush=True)
        rec, _ = svd_report(name, w, x)
        out["projections"][name] = rec
        print(
            f"    r=1024 energy={rec['energy_at_rank'].get('1024', float('nan')):.3f} "
            f"relF={rec['rel_f_at_rank'].get('1024', float('nan')):.3f} "
            f"relAct={rec['rel_act_at_rank'].get('1024', {}).get('mean', float('nan')):.3f}",
            flush=True,
        )

    print("  stacked [gate; up] SVD ...", flush=True)
    stacked = np.concatenate([wg, wu], axis=0)
    rec_st, (u_st, s_st, vt_st) = svd_report("gate_up_stack", stacked, x_h)
    out["projections"]["gate_up_stack"] = rec_st
    del stacked

    dense_layer = wg.size + wu.size + wd.size
    y_true = swiglu_down(wg, wu, wd, x_h)

    def e2e_err(wg_h, wu_h, wd_h) -> dict:
        y_h = swiglu_down(wg_h, wu_h, wd_h, x_h)
        num = np.linalg.norm(y_true - y_h, axis=0)
        den = np.linalg.norm(y_true, axis=0) + 1e-12
        ratios = num / den
        return {"mean": float(ratios.mean()), "p90": float(np.quantile(ratios, 0.90)), "max": float(ratios.max())}

    # --- 2:4 ---
    print("  2:4 prune ...", flush=True)
    wg24, wu24, wd24 = prune_2_4(wg), prune_2_4(wu), prune_2_4(wd)
    m24 = {
        "kind": "2:4 magnitude",
        "per_matrix": {
            "gate": {
                "rel_f": float(np.linalg.norm(wg - wg24) / np.linalg.norm(wg)),
                "rel_act": activation_err(wg, wg24, x_h),
                "params": int(wg.size // 2),
            },
            "up": {
                "rel_f": float(np.linalg.norm(wu - wu24) / np.linalg.norm(wu)),
                "rel_act": activation_err(wu, wu24, x_h),
                "params": int(wu.size // 2),
            },
            "down": {
                "rel_f": float(np.linalg.norm(wd - wd24) / np.linalg.norm(wd)),
                "rel_act": activation_err(wd, wd24, h_true),
                "params": int(wd.size // 2),
            },
        },
        "e2e_swiglu": e2e_err(wg24, wu24, wd24),
        "area": method_area(dense_layer / 2),
    }
    out["methods"]["sparse_2_4"] = m24
    del wg24, wu24, wd24

    # --- independent SVD at r=1024 and r=2048 ---
    for r in (1024, 2048):
        print(f"  independent SVD r={r} ...", flush=True)
        hats = {}
        per = {}
        params = 0
        for name, w, x in (("gate", wg, x_h), ("up", wu, x_h), ("down", wd, h_true)):
            u, s, vt = rsvd(w, r, seed=3)
            what = (u[:, :r] * s[:r].astype(np.float32)) @ vt[:r]
            hats[name] = what
            per[name] = {
                "rel_f": float(np.linalg.norm(w - what) / np.linalg.norm(w)),
                "rel_act": activation_err(w, what, x),
                "params": int(r * sum(w.shape)),
            }
            params += r * sum(w.shape)
        out["methods"][f"svd_indep_r{r}"] = {
            "kind": f"independent truncated SVD r={r} on gate, up, down",
            "per_matrix": per,
            "e2e_swiglu": e2e_err(hats["gate"], hats["up"], hats["down"]),
            "area": method_area(params),
        }
        del hats, u, s, vt

    # --- joint gate/up SVD, down dense or SVD ---
    for r in (1024, 2048):
        print(f"  joint gate+up SVD r={r} ...", flush=True)
        what = (u_st[:, :r] * s_st[:r].astype(np.float32)) @ vt_st[:r]
        wg_h, wu_h = what[: wg.shape[0]], what[wg.shape[0] :]
        per = {
            "gate": {
                "rel_f": float(np.linalg.norm(wg - wg_h) / np.linalg.norm(wg)),
                "rel_act": activation_err(wg, wg_h, x_h),
            },
            "up": {
                "rel_f": float(np.linalg.norm(wu - wu_h) / np.linalg.norm(wu)),
                "rel_act": activation_err(wu, wu_h, x_h),
            },
        }
        params_joint = r * (wg.shape[0] + wu.shape[0] + wg.shape[1])  # r*(2I+d)
        out["methods"][f"joint_gate_up_r{r}_down_dense"] = {
            "kind": f"shared-A SVD of [gate;up] r={r}, down unchanged",
            "per_matrix": per,
            "e2e_swiglu": e2e_err(wg_h, wu_h, wd),
            "area": method_area(params_joint + wd.size),
        }
        u_d, s_d, vt_d = rsvd(wd, r, seed=4)
        wd_h = (u_d[:, :r] * s_d[:r].astype(np.float32)) @ vt_d[:r]
        out["methods"][f"joint_gate_up_r{r}_down_svd_r{r}"] = {
            "kind": f"shared-A SVD r={r} on [gate;up] + independent SVD r={r} on down",
            "per_matrix": {
                **per,
                "down": {
                    "rel_f": float(np.linalg.norm(wd - wd_h) / np.linalg.norm(wd)),
                    "rel_act": activation_err(wd, wd_h, h_true),
                },
            },
            "e2e_swiglu": e2e_err(wg_h, wu_h, wd_h),
            "area": method_area(params_joint + r * sum(wd.shape)),
        }
        del what, wg_h, wu_h, wd_h, u_d, s_d, vt_d

    # --- Monarch ---
    for b in MONARCH_BLOCKS:
        print(f"  Monarch nblocks={b} ...", flush=True)
        per = {}
        params = 0
        hats = {}
        for name, w, x in (("gate", wg, x_h), ("up", wu, x_h), ("down", wd, h_true)):
            w1, w2, rel_f, rank = monarch_project(w, b)
            apply = lambda xx, _w1=w1, _w2=w2: monarch_apply(_w1, _w2, xx)
            per[name] = {
                "rel_f": rel_f,
                "rel_act": activation_err_factors(w, apply, x),
                "params": monarch_params(b, w.shape[0], w.shape[1]),
                "rank_per_block": rank,
                "nblocks": b,
            }
            params += per[name]["params"]
            hats[name] = apply
        y_h = hats["down"](silu(hats["gate"](x_h)) * hats["up"](x_h))
        num = np.linalg.norm(y_true - y_h, axis=0)
        den = np.linalg.norm(y_true, axis=0) + 1e-12
        ratios = num / den
        out["methods"][f"monarch_b{b}"] = {
            "kind": (
                f"rectangular Monarch nblocks={b}, rank=min(d,I)/b² per 4D slice "
                "(HazyResearch MonarchLinear parameter budget)"
            ),
            "per_matrix": per,
            "e2e_swiglu": {
                "mean": float(ratios.mean()),
                "p90": float(np.quantile(ratios, 0.90)),
                "max": float(ratios.max()),
            },
            "area": method_area(params),
            "note": (
                "User slide used 2/b × dense (50% at b=4). Actual MonarchLinear "
                "on 5120×17408 is min(d,I)·(d+I)/b ≈ 32% at b=4, 16% at b=8."
            ),
        }
        del hats, w1, w2

    out["dense_area"] = method_area(dense_layer)
    return out


def plot_spectra(proj: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {
        "gate": "#c23b22",
        "up": "#c9a227",
        "down": "#2c5f8a",
        "gate_up_stack": "#2d7a4f",
    }
    for name, rec in proj.items():
        if name == "down_isotropic" or "energy_curve" not in rec:
            continue
        ranks = rec["energy_curve_ranks"]
        ax.plot(ranks, rec["energy_curve"], label=name, color=colors.get(name, "#333"), lw=1.8)
    ax.axvline(1024, color="#999", ls="--", lw=0.8)
    ax.axvline(2048, color="#bbb", ls=":", lw=0.8)
    ax.set_xlabel("rank")
    ax.set_ylabel("captured energy  (Σ σ² / ||W||_F²)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, RSVD_K)
    ax.set_title("Qwen3.8-27B layer 0 FFN — SVD energy")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_methods(methods: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    keys = [
        "sparse_2_4",
        "svd_indep_r1024",
        "svd_indep_r2048",
        "joint_gate_up_r1024_down_dense",
        "joint_gate_up_r1024_down_svd_r1024",
        "monarch_b4",
        "monarch_b8",
    ]
    labels, e2e, mm2 = [], [], []
    for k in keys:
        if k not in methods:
            continue
        labels.append(k.replace("_", "\n"))
        e2e.append(methods[k]["e2e_swiglu"]["mean"])
        mm2.append(methods[k]["area"]["rom_mm2"])
    fig, ax1 = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, e2e, 0.36, color="#c23b22", label="SwiGLU e2e rel. act. err")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, mm2, 0.36, color="#2c5f8a", label="64L Q4 ROM mm²")
    ax2.axhline(RETICLE, color="#2d7a4f", ls="--", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("mean ||y−ŷ|| / ||y||  (layer 0, unit-RMS x)")
    ax2.set_ylabel("FFN via-ROM mm²")
    ax1.set_title("Post-hoc FFN structure — quality vs 28 nm Q4 area")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _load_layer(layer: int, weight_map: dict) -> dict[str, np.ndarray]:
    print(f"  load layer {layer} ...", flush=True)
    return {p: load_weight(mlp_name(layer, p), weight_map) for p in ("gate", "up", "down")}


def _pair_metrics(a: int, wa: dict, b: int, wb: dict) -> dict:
    entry = {"layers": [a, b], "matrices": {}}
    for p in ("gate", "up", "down"):
        entry["matrices"][p] = {
            "cosine_flat": cosine_flat(wa[p], wb[p]),
            "mean_row_cosine": mean_row_cosine(wa[p], wb[p]),
            "cka_rows": linear_cka(wa[p], wb[p]),
        }
    print(
        f"    L{a} vs L{b} down cosine={entry['matrices']['down']['cosine_flat']:.4f} "
        f"CKA={entry['matrices']['down']['cka_rows']:.4f}",
        flush=True,
    )
    return entry


def similarity(weight_map: dict, layer0: dict) -> dict:
    rec: dict = {"pairs": []}
    layer1 = _load_layer(1, weight_map)
    rec["pairs"].append(_pair_metrics(0, layer0, 1, layer1))
    del layer1
    layer62 = _load_layer(62, weight_map)
    layer63 = _load_layer(63, weight_map)
    rec["pairs"].append(_pair_metrics(62, layer62, 63, layer63))
    rec["pairs"].append(_pair_metrics(0, layer0, 62, layer62))
    rec["pairs"].append(_pair_metrics(0, layer0, 63, layer63))
    del layer62, layer63
    return rec


def _self_check() -> None:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((32, 64), dtype=np.float32)
    w1, w2, rel_f, _rank = monarch_project(m, 4)
    what = monarch_apply(w1, w2, np.eye(64, dtype=np.float32))
    err = float(np.linalg.norm(m - what) / np.linalg.norm(m))
    if abs(err - rel_f) > 0.02:
        raise RuntimeError(f"monarch roundtrip {err:.4f} vs rel_f {rel_f:.4f}")


def main() -> None:
    _self_check()
    ART.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    print("index ...", flush=True)
    idx = load_index()
    weight_map = idx["weight_map"]

    layer0 = _load_layer(PROBE_LAYER, weight_map)
    wg, wu, wd = layer0["gate"], layer0["up"], layer0["down"]
    assert wg.shape == (INTER, HIDDEN), wg.shape
    assert wu.shape == (INTER, HIDDEN), wu.shape
    assert wd.shape == (HIDDEN, INTER), wd.shape

    x_h = unit_rms_x(HIDDEN, N_X, seed=1)
    print("probe layer 0 ...", flush=True)
    probe = probe_layer(wg, wu, wd, x_h)

    print("adjacent-layer similarity ...", flush=True)
    sim = similarity(weight_map, layer0)
    del layer0, wg, wu, wd

    plot_spectra(probe["projections"], ART / "ffn-structure-svd.png")
    plot_methods(probe["methods"], ART / "ffn-structure-methods.png")

    summary = {
        "checkpoint": REPO,
        "layer": PROBE_LAYER,
        "x": {
            "kind": "unit-RMS Gaussian, n=128 (post-RMSNorm model of FFN input)",
            "note": "Not cached hidden states. Isotropic x; real residual stream is anisotropic, so this is a lower bound on activation error for a given Frobenius fit.",
        },
        "monarch_definition": {
            "nblocks": list(MONARCH_BLOCKS),
            "rank_per_slice": "min(out,in)/nblocks²",
            "params_one_matrix": "rank * nblocks * (out+in) = min(out,in)*(out+in)/nblocks",
            "matches": "HazyResearch MonarchLinear, not the slide's 2/nblocks * dense",
        },
        "area_cell": {
            "um2_per_weight_q4": BITS * CELL / EFF,
            "reticle_mm2": RETICLE,
        },
        "similarity": sim,
        "layer0": probe,
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    outp = ART / "ffn_structure_probe.json"
    outp.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {outp} in {summary['elapsed_sec']}s", flush=True)

    print("\n=== e2e SwiGLU rel act (layer 0) / 64L Q4 mm² ===", flush=True)
    for k, v in probe["methods"].items():
        e = v["e2e_swiglu"]["mean"]
        a = v["area"]
        print(
            f"  {k:40s}  e2e={e:.3f}  {a['rom_mm2']:7.1f} mm²  "
            f"{a['pct_reticle']:5.1f}% reticle  {a['compression']:.2f}×",
            flush=True,
        )


if __name__ == "__main__":
    main()
