"""Yosys (+ optional OpenLane) for the three spatial FFN column DUTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sky130_sweep import CLOCK_NS, F2_SCALE_7NM, SRAM_SCALE_7NM, run_openlane, yosys_synth  # noqa: E402

DUTS = (
    {
        "name": "ffn_col_serial_8x8_b4",
        "rtl": ROOT / "rtl" / "ffn_col_serial.v",
        "top": "ffn_col_serial",
        "cfg": ROOT / "openlane" / "ffn_col_serial.json",
        "tag": "xbar_col_serial",
        "kind": "serial_csd",
    },
    {
        "name": "ffn_rom_tap_8x8_b4",
        "rtl": ROOT / "rtl" / "ffn_rom_tap.v",
        "top": "ffn_rom_tap_reg",
        "cfg": ROOT / "openlane" / "ffn_rom_tap.json",
        "tag": "xbar_rom_tap",
        "kind": "rom_tap",
    },
    {
        "name": "ffn_rom_fetch_8x8_b4",
        "rtl": ROOT / "rtl" / "ffn_rom_fetch.v",
        "top": "ffn_rom_fetch",
        "cfg": ROOT / "openlane" / "ffn_rom_fetch.json",
        "tag": "xbar_rom_fetch",
        "kind": "rom_fetch",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnr", action="store_true")
    parser.add_argument("--only", default="", help="comma names to run (serial,tap,fetch)")
    args = parser.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    results = []
    for spec in DUTS:
        key = spec["kind"].replace("rom_", "").replace("serial_csd", "serial")
        if only and key not in only and spec["kind"] not in only:
            continue
        print(f"=== yosys {spec['top']} ===", flush=True)
        rpt = ROOT / "artifacts" / "eda" / f"yosys_{spec['tag']}.rpt"
        ys = yosys_synth(spec["rtl"], spec["top"], rpt)
        ys.pop("stat_text_tail", None)
        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            "top": spec["top"],
            "yosys": ys,
        }
        if args.pnr:
            print(f"=== openlane {spec['tag']} ===", flush=True)
            row["openlane"] = run_openlane(spec["cfg"], spec["tag"])
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "yosys"}, default=str), flush=True)

    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "clock_ns": CLOCK_NS,
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "note": (
            "Three 8x8 4-bit DUTs, same W as ffn_tile_8x8_b4_reg. "
            "rom_tap/rom_fetch are stdcell stand-ins, not via-ROM density. "
            "7nm F²=(7/130)^2."
        ),
        "duts": results,
    }
    out = ROOT / "artifacts" / "xbar_eda.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
