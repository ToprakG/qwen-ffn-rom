#!/usr/bin/env python3
"""Quantize Qwen3.5-0.8B layer-0 down_proj and write integer tensors + quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.load_ffn import TARGET_NAME, load_down_proj
from quant.quantize import QuantizedMatrix, matvec_int, quantize_per_row, quality_report

TILE = 128


def main() -> None:
    cache = ROOT / "quant" / "cache"
    out = ROOT / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    w = load_down_proj(cache)
    reports = {}
    for bits in (2, 3, 4):
        q = quantize_per_row(w, bits)
        np.save(out / f"W_int{bits}.npy", q.w_int)
        np.save(out / f"W_int{bits}_scale.npy", q.scale)
        (out / f"W_int{bits}_scale.json").write_text(
            json.dumps(
                {
                    "scheme": "symmetric_per_output_row",
                    "bits": bits,
                    "qmin": q.qmin,
                    "qmax": q.qmax,
                    "n_scales": int(q.scale.size),
                    "scale_mean": float(np.mean(q.scale)),
                }
            )
            + "\n"
        )
        reports[str(bits)] = quality_report(w, q)
        for t in (4, 8, 16, TILE):
            np.save(out / f"tile{t}_int{bits}.npy", q.w_int[:t, :t])
        tile_q = QuantizedMatrix(
            w_int=q.w_int[:TILE, :TILE],
            scale=q.scale[:TILE],
            bits=q.bits,
            qmin=q.qmin,
            qmax=q.qmax,
        )
        reports[str(bits)]["tile128"] = quality_report(w[:TILE, :TILE], tile_q, n_vec=64)

    rng = np.random.default_rng(0)
    w4 = np.load(out / "W_int4.npy")
    x = rng.integers(-128, 128, size=(TILE,), dtype=np.int32)
    y = matvec_int(w4[:TILE, :TILE], x)
    np.save(out / "ref_x_int8.npy", x.astype(np.int8))
    np.save(out / "ref_y_int.npy", y)

    payload = {
        "tensor": TARGET_NAME,
        "fp_shape": list(w.shape),
        "fp_mean": float(w.mean()),
        "fp_std": float(w.std()),
        "fp_absmax": float(abs(w).max()),
        "tile": TILE,
        "bits": reports,
        "ref_matvec": {
            "note": "y = W_int[0:128,0:128] @ x_int8  (int32 accumulate)",
            "x": "artifacts/ref_x_int8.npy",
            "y": "artifacts/ref_y_int.npy",
        },
    }
    (out / "quant_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    slim = _drop_hist(reports)
    print(json.dumps(slim, indent=2))
    print(f"wrote {out}")


def _drop_hist(obj):
    if isinstance(obj, dict):
        return {k: _drop_hist(v) for k, v in obj.items() if k != "int_hist"}
    return obj


if __name__ == "__main__":
    main()
