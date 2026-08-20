#!/usr/bin/env python3
"""Yosys (+ optional OpenLane) for the complete qwen_layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sky130_sweep import (  # noqa: E402
    CLOCK_NS,
    F2_SCALE_7NM,
    SRAM_SCALE_7NM,
    parse_openlane_metrics,
    run_openlane,
    yosys_synth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnr", action="store_true")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="parse an existing openlane/runs/qwen_layer without re-running PnR",
    )
    args = parser.parse_args()

    rtl = [
        ROOT / "rtl" / "add_tree_bal.v",
        ROOT / "rtl" / "rsqrt_lut.v",
        ROOT / "rtl" / "inv_rsqrt.v",
        ROOT / "rtl" / "rmsnorm.v",
        ROOT / "rtl" / "gated_delta_step.v",
        ROOT / "rtl" / "attn_decode.v",
        ROOT / "rtl" / "ffn_tile_16x16_b4_reg.v",
        ROOT / "rtl" / "qwen_layer.v",
    ]
    if args.ingest and (ROOT / "artifacts" / "qwen_layer_eda.json").exists():
        prev = json.loads((ROOT / "artifacts" / "qwen_layer_eda.json").read_text())
        ys = prev.get("yosys") or {}
        print("=== yosys qwen_layer (cached) ===", flush=True)
    else:
        rpt = ROOT / "artifacts" / "eda" / "yosys_qwen_layer.rpt"
        print("=== yosys qwen_layer ===", flush=True)
        ys = yosys_synth(rtl, "qwen_layer", rpt)
        ys.pop("stat_text_tail", None)

    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "clock_ns": CLOCK_NS,
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "dut": "qwen_layer",
        "hidden": 16,
        "heads": 4,
        "d": 4,
        "s_max": 8,
        "note": (
            "Complete decoder layer: 2× RMSNorm H=16, 4× gated_delta_step D=4, "
            "attn_decode D=4 S_MAX=8, 16×16 CSD FFN from 0.8B down_proj. "
            "7nm area is F²=(7/130)^2 on Sky130 instance area, not 7nm STA. "
            "28nm Fmax is unmeasured."
        ),
        "yosys": ys,
    }
    if ys.get("area_um2"):
        payload["area_7nm_f2_um2"] = ys["area_um2"] * F2_SCALE_7NM
        payload["area_7nm_sram_um2"] = ys["area_um2"] * SRAM_SCALE_7NM

    if args.pnr:
        print("=== openlane qwen_layer ===", flush=True)
        cfg = ROOT / "openlane" / "qwen_layer.json"
        payload["openlane"] = run_openlane(cfg, "qwen_layer")
    elif args.ingest:
        run_dir = ROOT / "openlane" / "runs" / "qwen_layer"
        payload["openlane"] = parse_openlane_metrics(run_dir)
        payload["openlane"]["tag"] = "qwen_layer"
        payload["openlane"]["note"] = (
            "Ingested post-route STA. Flow quit in Magic LEF/antenna after TritonRoute DRC=0. "
            "Magic DRC/LVS unmeasured."
        )

    ol = payload.get("openlane") or {}
    if ol.get("area_um2"):
        payload["area_7nm_f2_um2"] = ol["area_um2"] * F2_SCALE_7NM
        payload["area_source"] = ol.get("metrics_source") or "openlane"
    if ol.get("fmax_mhz_from_ws"):
        payload["fmax_mhz_sky130"] = ol["fmax_mhz_from_ws"]
        payload["fmax_source"] = "openlane_postroute_ws"

    out = ROOT / "artifacts" / "qwen_layer_eda.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    slim = {k: v for k, v in payload.items() if k not in ("yosys",)}
    print(json.dumps(slim, default=str, indent=2), flush=True)


if __name__ == "__main__":
    main()
