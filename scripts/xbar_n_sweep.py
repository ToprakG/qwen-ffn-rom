#!/usr/bin/env python3
"""N-sweep: word-parallel CSD vs bit-serial CSD until STA dies.

STA is ABC cell delay (stime) plus a Sky130 wire factor calibrated on the
signed-off 8x8 tile (OpenLane path / ABC path). Optional OpenLane PnR for
N<=16. Same integer golden y = W_int @ x.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pe_xbar.emit import write_all  # noqa: E402
from pe_xbar.weights import load_w  # noqa: E402
from rtl_gen.emit_tile import write_tile  # noqa: E402
from sky130_sweep import (  # noqa: E402
    CLOCK_NS,
    LIBERTY,
    parse_yosys_stat,
    run_openlane,
)

# Calibrated on ffn_tile_8x8_b4_reg: OpenLane path 13.32 ns / ABC 3.045 ns.
WIRE_FACTOR = 4.37
ABC_SCRIPT = (
    "+strash;ifraig;scorr;dc2;retime,D,{{D}};strash;dch,-f;"
    "map,-M,1,{{D}};topo;upsize,{{D}};dnsize,{{D}};stime,-p"
)


def yosys_sta(rtl: Path, top: str, rpt: Path, mapped: Path) -> dict:
    if not LIBERTY.exists():
        raise SystemExit(f"missing liberty {LIBERTY}")
    ys = f"""
read_verilog -sv {rtl.as_posix()}
hierarchy -check -top {top}
proc; flatten
opt_expr; opt_clean
synth -top {top}
dfflibmap -liberty {LIBERTY.as_posix()}
abc -liberty {LIBERTY.as_posix()} -script {ABC_SCRIPT}
opt_clean
stat -liberty {LIBERTY.as_posix()}
write_verilog -noattr {mapped.as_posix()}
"""
    rpt.parent.mkdir(parents=True, exist_ok=True)
    mapped.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["yosys", "-ql", str(rpt.with_suffix(".log")), "-p", ys],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log = rpt.with_suffix(".log").read_text() if rpt.with_suffix(".log").exists() else ""
    combined = log + "\n" + proc.stdout + "\n" + proc.stderr
    rpt.write_text(combined)
    if proc.returncode != 0:
        raise SystemExit(f"yosys failed for {rtl.name}:\n{proc.stderr[-2000:]}")
    stats = parse_yosys_stat(combined)
    m = re.search(r"Delay\s*=\s*([0-9.]+)\s*ps", combined)
    delay_ps = float(m.group(1)) if m else None
    delay_ns = delay_ps / 1000.0 if delay_ps is not None else None
    pred_ns = delay_ns * WIRE_FACTOR if delay_ns is not None else None
    stats.update(
        {
            "rtl": str(rtl.relative_to(ROOT)),
            "top": top,
            "abc_delay_ps": delay_ps,
            "abc_delay_ns": delay_ns,
            "abc_fmax_mhz": (1000.0 / delay_ns) if delay_ns and delay_ns > 0 else None,
            "pred_pnr_path_ns": pred_ns,
            "pred_pnr_fmax_mhz": (1000.0 / pred_ns) if pred_ns and pred_ns > 0 else None,
            "sta_dies_50ns": bool(pred_ns is not None and pred_ns > CLOCK_NS),
            "wire_factor": WIRE_FACTOR,
            "mapped": str(mapped.relative_to(ROOT)),
        }
    )
    return stats


def write_serial_ol(n: int, rtl: Path) -> Path:
    cfg = {
        "DESIGN_NAME": "ffn_col_serial",
        "VERILOG_FILES": f"dir::{os.path.relpath(rtl, ROOT / 'openlane')}",
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": CLOCK_NS,
        "FP_SIZING": "relative",
        "FP_CORE_UTIL": 20,
        "PL_TARGET_DENSITY_PCT": 30,
        "MAX_FANOUT_CONSTRAINT": 16,
        "RUN_LINTER": False,
        "QUIT_ON_SYNTH_CHECKS": False,
    }
    path = ROOT / "openlane" / f"ffn_col_serial_{n}x{n}_b4.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def write_tile_ol(n: int, rtl: Path) -> Path:
    cfg = {
        "DESIGN_NAME": "ffn_tile_reg",
        "VERILOG_FILES": f"dir::{os.path.relpath(rtl, ROOT / 'openlane')}",
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": CLOCK_NS,
        "FP_SIZING": "relative",
        "FP_CORE_UTIL": 20,
        "PL_TARGET_DENSITY_PCT": 30,
        "MAX_FANOUT_CONSTRAINT": 16,
        "RUN_LINTER": False,
        "QUIT_ON_SYNTH_CHECKS": False,
    }
    path = ROOT / "openlane" / f"tile_{n}x{n}_b4.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def merge_sweep_points(new_points: list[dict], path: Path) -> list[dict]:
    """Keep prior N/style rows (equiv, OpenLane) when a partial re-run writes."""
    by_key: dict[tuple, dict] = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = {}
        for p in old.get("points") or []:
            by_key[(p.get("n"), p.get("style"))] = p
    for p in new_points:
        key = (p.get("n"), p.get("style"))
        prev = by_key.get(key) or {}
        merged = dict(prev)
        merged.update(p)
        if "openlane" not in merged and "openlane" in prev:
            merged["openlane"] = prev["openlane"]
            if "tok_s_pnr" not in merged and "tok_s_pnr" in prev:
                merged["tok_s_pnr"] = prev["tok_s_pnr"]
        if "equiv" not in merged and "equiv" in prev:
            merged["equiv"] = prev["equiv"]
        by_key[key] = merged
    order: list[dict] = []
    ns = sorted({n for n, _ in by_key if n is not None})
    for n in ns:
        for style in ("wordpar_csd", "serial_csd"):
            if (n, style) in by_key:
                order.append(by_key[(n, style)])
        for (nn, style), row in by_key.items():
            if nn == n and style not in ("wordpar_csd", "serial_csd"):
                order.append(row)
    return order


def run_equiv(n: int, dut: str) -> dict | None:
    env = os.environ.copy()
    cmd = [
        "make",
        "-C",
        str(ROOT / "tb"),
        "-f",
        "Makefile.xbar",
        f"XBAR_DUT={dut}",
        f"FFN_TILE_N={n}",
    ]
    print(f"=== equiv {dut} n={n} ===", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=env)
    if p.returncode != 0:
        return {"status": "FAIL", "dut": dut, "n": n}
    name = f"{dut}_{n}x{n}_b4"
    path = ROOT / "artifacts" / f"{name}_equiv.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"status": "PASS", "dut": dut, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", default="8,16,32")
    parser.add_argument("--pnr-max-n", type=int, default=16)
    parser.add_argument("--pnr", action="store_true")
    parser.add_argument("--equiv", action="store_true")
    args = parser.parse_args()
    tiles = [int(x) for x in args.tiles.split(",") if x]

    points = []
    for n in tiles:
        w = load_w(n, 4)
        print(f"=== gen {n}x{n} nonzero={int((w != 0).sum())}/{w.size} ===", flush=True)
        if n == 8:
            serial_rtl = ROOT / "rtl" / "ffn_col_serial.v"
            tile_rtl = ROOT / "rtl" / "ffn_tile_8x8_b4_reg.v"
            if not serial_rtl.exists():
                write_all(ROOT / "rtl", w, 4, ("serial",))
        else:
            write_all(ROOT / "rtl", w, 4, ("serial",))
            serial_rtl = ROOT / "rtl" / f"ffn_col_serial_{n}x{n}_b4.v"
            tile_rtl = ROOT / "rtl" / f"ffn_tile_{n}x{n}_b4_reg.v"
            write_tile(
                tile_rtl,
                w,
                4,
                in_w=8,
                module="ffn_tile",
                registered=True,
                reg_module="ffn_tile_reg",
            )

        for style, rtl, top, cycles in (
            ("wordpar_csd", tile_rtl, "ffn_tile_reg", 1),
            ("serial_csd", serial_rtl, "ffn_col_serial", 9),
        ):
            print(f"=== yosys+sta {style} {n}x{n} ===", flush=True)
            rpt = ROOT / "artifacts" / "eda" / f"sta_{style}_{n}x{n}.rpt"
            mapped = ROOT / "artifacts" / "mapped" / f"{style}_{n}x{n}.v"
            sta = yosys_sta(rtl, top, rpt, mapped)
            row = {
                "n": n,
                "bits": 4,
                "style": style,
                "top": top,
                "cycles_per_token": cycles,
                "yosys": {k: sta[k] for k in ("cell_count", "area_um2", "sequential_cells") if k in sta},
                "sta": {
                    "abc_delay_ns": sta.get("abc_delay_ns"),
                    "abc_fmax_mhz": sta.get("abc_fmax_mhz"),
                    "pred_pnr_path_ns": sta.get("pred_pnr_path_ns"),
                    "pred_pnr_fmax_mhz": sta.get("pred_pnr_fmax_mhz"),
                    "sta_dies_50ns": sta.get("sta_dies_50ns"),
                    "wire_factor": WIRE_FACTOR,
                },
            }
            tok = None
            if sta.get("pred_pnr_fmax_mhz") and cycles:
                tok = (sta["pred_pnr_fmax_mhz"] * 1e6) / cycles
            row["tok_s_pred"] = tok

            if args.equiv and style == "serial_csd":
                row["equiv"] = run_equiv(n, "ffn_col_serial")

            do_pnr = args.pnr and n <= args.pnr_max_n and n > 8
            if do_pnr:
                print(f"=== openlane {style} {n}x{n} ===", flush=True)
                if style == "serial_csd":
                    cfg = write_serial_ol(n, rtl)
                    tag = f"xbar_serial_{n}x{n}"
                else:
                    cfg = write_tile_ol(n, rtl)
                    tag = f"tile_{n}x{n}_b4"
                row["openlane"] = run_openlane(cfg, tag)
                ol = row["openlane"]
                if ol.get("fmax_mhz_from_ws") and cycles:
                    row["tok_s_pnr"] = (ol["fmax_mhz_from_ws"] * 1e6) / cycles

            points.append(row)
            slim = {k: v for k, v in row.items() if k not in ("yosys",)}
            print(json.dumps(slim, default=str), flush=True)

    out = ROOT / "artifacts" / "xbar_n_sweep.json"
    points = merge_sweep_points(points, out)
    payload = {
        "clock_ns": CLOCK_NS,
        "wire_factor": WIRE_FACTOR,
        "wire_factor_note": (
            "pred_pnr_path = abc_delay * 4.37 from 8x8 CSD tile "
            "(OpenLane 13.32 ns / ABC 3.045 ns). sta_dies_50ns uses that prediction."
        ),
        "points": points,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
