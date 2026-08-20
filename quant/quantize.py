"""Symmetric per-output-channel quantizer and integer mat-vec reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizedMatrix:
    w_int: np.ndarray
    scale: np.ndarray
    bits: int
    qmin: int
    qmax: int

    def dequant(self) -> np.ndarray:
        return self.w_int.astype(np.float32) * self.scale[:, None].astype(np.float32)


def quantize_per_row(w: np.ndarray, bits: int) -> QuantizedMatrix:
    """Symmetric uniform quant, one scale per output row (standard for GEMV)."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    amax = np.max(np.abs(w.astype(np.float32)), axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    q = np.clip(np.rint(w.astype(np.float32) / scale), qmin, qmax).astype(np.int8)
    return QuantizedMatrix(w_int=q, scale=scale.reshape(-1), bits=bits, qmin=qmin, qmax=qmax)


def matvec_int(w_int: np.ndarray, x_int: np.ndarray) -> np.ndarray:
    """Equivalence baseline: y = W_int @ x_int in int32."""
    return w_int.astype(np.int32) @ x_int.astype(np.int32)


def matvec_dequant(q: QuantizedMatrix, x: np.ndarray) -> np.ndarray:
    return q.dequant() @ x.astype(np.float32)


def quality_report(w: np.ndarray, q: QuantizedMatrix, n_vec: int = 256, seed: int = 0) -> dict:
    wd = q.dequant()
    wf = w.astype(np.float32)
    err = wd - wf
    mse = float(np.mean(err * err))
    var = float(np.mean(wf * wf))
    snr_db = 10.0 * np.log10(var / mse) if mse > 0 else float("inf")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((w.shape[1], n_vec), dtype=np.float32)
    y_fp = wf @ x
    y_q = wd @ x
    num = np.sum(y_fp * y_q, axis=0)
    den = np.linalg.norm(y_fp, axis=0) * np.linalg.norm(y_q, axis=0)
    cosine = num / np.clip(den, 1e-12, None)
    rel = np.linalg.norm(y_q - y_fp, axis=0) / np.clip(np.linalg.norm(y_fp, axis=0), 1e-12, None)
    hist = {int(k): int(v) for k, v in zip(*np.unique(q.w_int, return_counts=True))}
    return {
        "bits": q.bits,
        "scale_mean": float(np.mean(q.scale)),
        "scale_min": float(np.min(q.scale)),
        "scale_max": float(np.max(q.scale)),
        "qmin": q.qmin,
        "qmax": q.qmax,
        "shape": list(w.shape),
        "weight_mse": mse,
        "weight_mae": float(np.mean(np.abs(err))),
        "weight_max_abs_err": float(np.max(np.abs(err))),
        "weight_snr_db": snr_db,
        "matvec_cosine_mean": float(np.mean(cosine)),
        "matvec_cosine_min": float(np.min(cosine)),
        "matvec_rel_l2_mean": float(np.mean(rel)),
        "matvec_rel_l2_p95": float(np.quantile(rel, 0.95)),
        "int_hist": hist,
        "zero_frac": float(np.mean(q.w_int == 0)),
    }
