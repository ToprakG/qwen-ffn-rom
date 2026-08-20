#!/usr/bin/env python3
"""Quality-gate int4 KV vs fp32 / int8-KV attention (no RTL)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from quant.attn_online_int import (  # noqa: E402
    attn_fp32,
    attn_online_int,
    pack_int4_asr,
    quant_int4,
)

SHIFT = 8


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    return float(np.dot(a, b) / (na * nb))


def attn_int8_mac(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Old RTL skeleton: no softmax. For reference only."""
    scores = (K.astype(np.int64) @ q.astype(np.int64)) >> SHIFT
    return (V.astype(np.int64).T @ scores) >> SHIFT


def main() -> None:
    rng = np.random.default_rng(11)
    d, s, p, n = 64, 128, 8, 24
    rows = []
    cos_i4_fp = []
    cos_i4deq_fp = []
    cos_i8_fp = []
    cos_ol_fp = []
    for i in range(n):
        qf = rng.normal(0, 1, size=d)
        Kf = rng.normal(0, 1, size=(s, d))
        Vf = rng.normal(0, 1, size=(s, d))
        ref = attn_fp32(qf, Kf, Vf)

        q8 = np.clip(np.round(qf * 32), -128, 127).astype(np.int64)
        K8 = np.clip(np.round(Kf * 32), -128, 127).astype(np.int64)
        V8 = np.clip(np.round(Vf * 32), -128, 127).astype(np.int64)
        K4s, sk = quant_int4(Kf)
        V4s, sv = quant_int4(Vf)
        deq = attn_fp32(qf, K4s.astype(np.float64) * sk, V4s.astype(np.float64) * sv)

        K4 = pack_int4_asr(K8)
        V4 = pack_int4_asr(V8)
        ol4 = attn_online_int(q8, K4, V4, p)
        ol8 = attn_online_int(q8, K8 >> 4, V8 >> 4, p)

        c_deq = cosine(ref, deq)
        c_ol = cosine(ref, ol4.astype(np.float64))
        c_i8 = cosine(ref, attn_fp32(q8 / 32.0, K8 / 32.0, V8 / 32.0))
        cos_i4deq_fp.append(c_deq)
        cos_ol_fp.append(c_ol)
        cos_i8_fp.append(c_i8)
        cos_i4_fp.append(cosine(deq, ol4.astype(np.float64)))
        rows.append({"i": i, "cos_int4_dequant_fp32": c_deq, "cos_online_int4_fp32": c_ol})

    mean_deq = float(np.mean(cos_i4deq_fp))
    mean_ol = float(np.mean(cos_ol_fp))
    mean_i8 = float(np.mean(cos_i8_fp))
    # int4 KV dequant vs fp32 is the quality that matters for the cache format.
    # Threshold 0.98 is typical for int4 KV in decode (well-studied, usually cheap).
    status = "PASS" if mean_deq >= 0.98 else "FAIL"
    payload = {
        "gate": "int4 KV vs fp32 softmax attention, cosine >= 0.98",
        "status": status,
        "n": n,
        "d": d,
        "S": s,
        "mean_cosine_int4_dequant_fp32": mean_deq,
        "min_cosine_int4_dequant_fp32": float(np.min(cos_i4deq_fp)),
        "mean_cosine_int8_qkv_fp32": mean_i8,
        "mean_cosine_online_int4_fp32": mean_ol,
        "note": (
            "Dequant int4 KV (per-channel symmetric along D) vs fp32 is the cache-format gate. "
            "Online integer softmax is a separate LUT/shift approx; it is bit-exact to RTL, "
            "not to fp32. Threshold 0.98 on dequant cosine."
        ),
        "threshold": 0.98,
        "rows": rows,
    }
    out = ROOT / "artifacts" / "attn_int4_quality.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{status} int4-dequant cosine={mean_deq:.4f}  online-int4 cosine={mean_ol:.4f}  int8 cosine={mean_i8:.4f}")
    print(f"wrote {out}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
