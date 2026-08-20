"""Yosys (+ optional OpenLane) for the hybrid layer stub."""

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
    run_openlane,
    yosys_synth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnr", action="store_true")
    args = parser.parse_args()

    rtl = [
        ROOT / "rtl" / "gated_delta_step.v",
        ROOT / "rtl" / "ffn_tile_8x8_b4_reg.v",
        ROOT / "rtl" / "hybrid_layer_stub.v",
    ]
    rpt = ROOT / "artifacts" / "eda" / "yosys_hybrid_layer.rpt"
    print("=== yosys hybrid_layer_stub ===", flush=True)
    ys = yosys_synth(rtl, "hybrid_layer_stub", rpt)
    ys.pop("stat_text_tail", None)

    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "clock_ns": CLOCK_NS,
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "dut": "hybrid_layer_stub",
        "d": 4,
        "n": 8,
        "bits": 4,
        "note": (
            "One-layer decode stub: gated_delta_d4 + ffn_tile_8x8_b4 on a shared "
            "int8 x. q=k=v=x[0:3]. FFN runs in parallel with the mixer. "
            "cycles/token and tok/s are the layer, not per-PE FFN."
        ),
        "yosys": ys,
    }
    if ys.get("area_um2"):
        payload["area_7nm_f2_um2"] = ys["area_um2"] * F2_SCALE_7NM
        payload["area_7nm_sram_um2"] = ys["area_um2"] * SRAM_SCALE_7NM

    if args.pnr:
        print("=== openlane hybrid_layer_stub ===", flush=True)
        cfg = ROOT / "openlane" / "hybrid_layer_stub.json"
        payload["openlane"] = run_openlane(cfg, "hybrid_layer_stub")

    out = ROOT / "artifacts" / "hybrid_layer_eda.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "yosys"}, default=str), flush=True)


if __name__ == "__main__":
    main()
