#!/usr/bin/env python3
"""Quality-gate Newton rsqrt RMSNorm vs restoring integer and fp32 (no RTL)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.layer_int import restoring_div_u32, restoring_isqrt  # noqa: E402
from quant.rsqrt_int import inv_rsqrt_q16, rmsnorm_inv, rmsnorm_nr  # noqa: E402

THRESH_COS = 0.99
THRESH_ABS = 1


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    return float(np.dot(a, b) / (na * nb))


def restoring_inv(ssq: int) -> int:
    r = max(restoring_isqrt(max(int(ssq), 1)), 1)
    return restoring_div_u32(1 << 16, r)


def main() -> None:
    rng = np.random.default_rng(7)
    rows = []
    ok = True
    for h, n in ((8, 400), (16, 400)):
        max_dy = 0
        n_diff = 0
        cs = []
        for _ in range(n):
            x = rng.integers(-128, 128, size=h, dtype=np.int64)
            w = rng.integers(-128, 128, size=h, dtype=np.int64)
            ssq = int((x * x).sum())
            y_nr = rmsnorm_nr(x, w)
            y_r = rmsnorm_inv(x, w, restoring_inv(ssq))
            dy = int(np.max(np.abs(y_nr - y_r)))
            max_dy = max(max_dy, dy)
            n_diff += int(np.any(y_nr != y_r))
            y_fp = x.astype(np.float64) * w.astype(np.float64) / math.sqrt(max(ssq, 1))
            cs.append(cosine(y_nr, y_fp))
        mean_c = float(np.mean(cs))
        min_c = float(np.min(cs))
        h_ok = max_dy <= THRESH_ABS and mean_c >= THRESH_COS
        ok = ok and h_ok
        rows.append({
            "h": h,
            "n": n,
            "max_abs_vs_restoring": max_dy,
            "frac_diff_restoring": n_diff / n,
            "mean_cosine_fp32": mean_c,
            "min_cosine_fp32": min_c,
            "pass": h_ok,
        })
    rec = {
        "gate": f"NR RMSNorm |y-y_restoring|<= {THRESH_ABS} and mean cosine vs fp32 >= {THRESH_COS} at H=8/16",
        "status": "PASS" if ok else "FAIL",
        "threshold_abs": THRESH_ABS,
        "threshold_cosine": THRESH_COS,
        "rows": rows,
        "note": (
            "NR matches 1/sqrt(ssq) more closely than floor(sqrt) then divide; "
            "int8 output stays within 1 of restoring at DUT widths. "
            "H=5120 Q16 inv collapses both paths (not this gate)."
        ),
    }
    out = ROOT / "artifacts" / "rsqrt_quality.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(
        f"{rec['status']}  "
        + "  ".join(
            f"H={r['h']} max|dy|={r['max_abs_vs_restoring']} cos={r['mean_cosine_fp32']:.4f}"
            for r in rows
        )
    )
    print(f"wrote {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
