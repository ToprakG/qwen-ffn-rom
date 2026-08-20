"""Yosys (+ optional OpenLane) for the Gated DeltaNet PE."""

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
    parser.add_argument("--d", type=int, default=4, choices=(4, 16, 128))
    parser.add_argument("--mac", type=int, default=1, choices=(1, 16))
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    d = args.d
    mac = args.mac

    if args.fused:
        if d not in (16, 128):
            d = 16
        rtl = [
            ROOT / "rtl" / "delta_s_col_ram.v",
            ROOT / "rtl" / "add_tree_bal.v",
            ROOT / "rtl" / "gated_delta_fused.v",
            ROOT / "rtl" / f"gated_delta_d{d}_fused.v",
        ]
        top = f"gated_delta_d{d}_fused"
        tag = top
        cfg = ROOT / "openlane" / "gated_delta_d16_fused.json"
        mac = d
    elif mac > 1:
        rtl: Path | list[Path] = [
            ROOT / "rtl" / "gated_delta_dpar.v",
            ROOT / "rtl" / "gated_delta_d16_par.v",
        ]
        top = "gated_delta_d16_par"
        tag = "gated_delta_d16_par"
        cfg = ROOT / "openlane" / "gated_delta_d16_par.json"
        d = 16
    elif d == 4:
        rtl = ROOT / "rtl" / "gated_delta_step.v"
        top = "gated_delta_step"
        tag = "gated_delta_d4"
        cfg = ROOT / "openlane" / "gated_delta_d4.json"
    else:
        rtl = [ROOT / "rtl" / "gated_delta_step.v", ROOT / "rtl" / "gated_delta_d16.v"]
        top = "gated_delta_d16"
        tag = "gated_delta_d16"
        cfg = ROOT / "openlane" / "gated_delta_d16.json"

    rpt = ROOT / "artifacts" / "eda" / f"yosys_{tag}.rpt"
    print(f"=== yosys {top} D={d} ===", flush=True)
    ys = yosys_synth(rtl, top, rpt, abc_fast=bool(args.fused))
    ys.pop("stat_text_tail", None)

    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "clock_ns": CLOCK_NS,
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "dut": top,
        "d": d,
        "mac": mac,
        "note": (
            f"Integer Gated DeltaNet PE, D={d}, {mac} MAC-lane(s) + D×D state. "
            + (
                "Column-stream fused update+readout; cycles D+2. 1-issue/cycle. "
                "Yosys synth-internal ABC (fraig) did not finish on the combo 4-mul EX; "
                "mapped with synth -noabc + abc strash;&nf. Area is liberty-mapped, not post-route. "
                if args.fused
                else (
                    "D-wide inner product; cycles ~O(D). Do not unroll D×D. "
                    if mac > 1
                    else "Work is O(D^2) cycles/token, independent of sequence length. Do not unroll. "
                )
            )
            + "7nm F²=(7/130)^2."
        ),
        "yosys": ys,
    }
    if ys.get("area_um2"):
        payload["area_7nm_f2_um2"] = ys["area_um2"] * F2_SCALE_7NM
        payload["area_7nm_sram_um2"] = ys["area_um2"] * SRAM_SCALE_7NM

    if not args.pnr and (ys.get("cell_count") or 0) > 40000:
        payload["pnr_skipped"] = (
            f"{ys.get('cell_count')} Yosys cells. Skip OpenLane by default "
            "(1-MAC D=16 was 21k). make pnr-delta-fused / pnr-delta-dpar / --pnr to force."
        )

    if args.pnr:
        if args.fused and d != 16:
            payload["pnr_skipped"] = "PnR the D=16 fused DUT; D=128 is sim+Yosys only."
        else:
            print(f"=== openlane {tag} ===", flush=True)
            payload["openlane"] = run_openlane(cfg, tag)

    if args.fused:
        out = ROOT / "artifacts" / f"delta_d{d}_fused_eda.json"
    elif mac > 1:
        out = ROOT / "artifacts" / "delta_dpar16_eda.json"
    elif d == 4:
        out = ROOT / "artifacts" / "delta_d4_eda.json"
    else:
        out = ROOT / "artifacts" / "delta_d16_eda.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "yosys"}, default=str), flush=True)


if __name__ == "__main__":
    main()
