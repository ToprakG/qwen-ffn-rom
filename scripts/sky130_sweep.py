"""Sky130 Yosys (+ optional OpenLane) sweep over tile size and bit-width."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.load_ffn import load_down_proj
from quant.quantize import quantize_per_row
from rtl_gen.emit_tile import write_tile

LIBERTY = (
    Path(os.environ.get("PDK_ROOT", Path.home() / ".volare"))
    / "sky130A"
    / "libs.ref"
    / "sky130_fd_sc_hd"
    / "lib"
    / "sky130_fd_sc_hd__tt_025C_1v80.lib"
)
CLOCK_NS = 50.0
# Dennard / F² area scale 130 nm → 7 nm. SRAM-cell ratio is the conservative alt.
F2_SCALE_7NM = (7.0 / 130.0) ** 2
SRAM_SCALE_7NM = 0.027 / 1.896


def gen_rtl(n: int, bits: int) -> Path:
    """Tile-local per-row quant so 2/3/4-bit sweep the same fp block, not a sparse corner."""
    cache = ROOT / "quant" / "cache"
    w = load_down_proj(cache)[:n, :n]
    q = quantize_per_row(w, bits)
    out = ROOT / "rtl" / f"ffn_tile_{n}x{n}_b{bits}_reg.v"
    write_tile(out, q.w_int, bits, in_w=8, module="ffn_tile", registered=True, reg_module="ffn_tile_reg")
    np.save(ROOT / "artifacts" / f"tile{n}_local_int{bits}.npy", q.w_int)
    nz = int(np.count_nonzero(q.w_int))
    print(f"  local-quant {n}x{n} {bits}-bit nonzero={nz}/{q.w_int.size}", flush=True)
    return out


def parse_yosys_stat(text: str) -> dict:
    cells = None
    area = None
    m = re.search(r"Number of cells:\s+([0-9]+)", text)
    if m:
        cells = int(m.group(1))
    if cells is None:
        m = re.search(r"^\s+(\d+)\s+[0-9.E+-]+\s+cells\s*$", text, re.M)
        if m:
            cells = int(m.group(1))
    m = re.search(r"Chip area for (?:module|top module) '[^']+':\s+([0-9.]+)", text)
    if m:
        area = float(m.group(1))
    seq = 0
    for kind, pat in (
        ("dff", r"sky130_fd_sc_hd__dfxtp_\d+\s+(\d+)"),
        ("dffe", r"sky130_fd_sc_hd__dfrtp_\d+\s+(\d+)"),
    ):
        seq += sum(int(x) for x in re.findall(pat, text))
    seq += sum(int(x) for x in re.findall(r"sky130_fd_sc_hd__df\w+_\d+\s+(\d+)", text))
    return {
        "cell_count": cells,
        "area_um2": area,
        "sequential_cells": seq,
        "stat_text_tail": "\n".join(text.strip().splitlines()[-40:]),
    }


def yosys_synth(
    rtl: Path | list[Path],
    top: str,
    rpt: Path,
    *,
    abc_fast: bool = False,
) -> dict:
    if not LIBERTY.exists():
        raise SystemExit(f"missing liberty {LIBERTY}")
    files = rtl if isinstance(rtl, list) else [rtl]
    reads = "\n".join(f"read_verilog -sv {p.as_posix()}" for p in files)
    abc_script = rpt.parent / "abc_strash_nf.script"
    if abc_fast:
        abc_script.write_text("strash\n&get -n\n&nf\n&put\nstime\n")
        synth = f"synth -top {top} -noabc"
        abc = f"abc -liberty {LIBERTY.as_posix()} -script {abc_script.as_posix()}"
    else:
        synth = f"synth -top {top}"
        abc = f"abc -liberty {LIBERTY.as_posix()}"
    ys = f"""
{reads}
hierarchy -check -top {top}
proc; flatten
opt_expr; opt_clean
{synth}
dfflibmap -liberty {LIBERTY.as_posix()}
{abc}
opt_clean
stat -liberty {LIBERTY.as_posix()}
"""
    rpt.parent.mkdir(parents=True, exist_ok=True)
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
        label = files[0].name if files else "rtl"
        raise SystemExit(f"yosys failed for {label}:\n{proc.stderr[-2000:]}")
    stats = parse_yosys_stat(combined)
    mdel = re.search(r"Delay\s*=\s*([0-9.]+)\s*ps", combined)
    delay_ps = float(mdel.group(1)) if mdel else None
    delay_ns = delay_ps / 1000.0 if delay_ps is not None else None
    stats["rtl"] = (
        [str(p.relative_to(ROOT)) for p in files] if len(files) > 1 else str(files[0].relative_to(ROOT))
    )
    stats["top"] = top
    stats["liberty"] = str(LIBERTY)
    stats["clock_ns"] = CLOCK_NS
    stats["abc"] = "strash_dc2_map" if abc_fast else "default"
    stats["abc_delay_ps"] = delay_ps
    stats["abc_delay_ns"] = delay_ns
    stats["abc_fmax_mhz"] = (1000.0 / delay_ns) if delay_ns and delay_ns > 0 else None
    return stats


def write_ol_config(n: int, bits: int, rtl: Path) -> Path:
    cfg_dir = ROOT / "openlane"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(rtl, cfg_dir)
    cfg = {
        "DESIGN_NAME": "ffn_tile_reg",
        "VERILOG_FILES": f"dir::{rel}",
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": CLOCK_NS,
        "FP_SIZING": "relative",
        "FP_CORE_UTIL": 20,
        "PL_TARGET_DENSITY_PCT": 30,
        "MAX_FANOUT_CONSTRAINT": 16,
        "RUN_LINTER": False,
        "QUIT_ON_SYNTH_CHECKS": False,
    }
    path = cfg_dir / f"tile_{n}x{n}_b{bits}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def pick_metric(d: dict, *keys: str):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    for k, v in d.items():
        if any(k.startswith(p) for p in keys) and v is not None:
            return v
    return None


def parse_openlane_metrics(run_dir: Path, period_ns: float | None = None) -> dict:
    metrics_path = run_dir / "final" / "metrics.json"
    source = "final/metrics.json"
    m: dict = {}
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
    else:
        sta = run_dir / "50-openroad-stapostpnr" / "state_out.json"
        if not sta.exists():
            return {"error": f"missing {metrics_path}"}
        m = json.loads(sta.read_text()).get("metrics") or {}
        source = "50-openroad-stapostpnr/state_out.json"
        trdrc = run_dir / "42-checker-trdrc" / "state_out.json"
        if trdrc.exists():
            tm = json.loads(trdrc.read_text()).get("metrics") or json.loads(trdrc.read_text())
            if "route__drc_errors" in tm:
                m["route__drc_errors"] = tm["route__drc_errors"]
    ws = pick_metric(m, "timing__setup__ws")
    period = float(period_ns) if period_ns is not None else CLOCK_NS
    fmax_mhz = None
    if isinstance(ws, (int, float)) and ws != float("inf"):
        path_ns = period - float(ws)
        if path_ns > 0:
            fmax_mhz = 1000.0 / path_ns
    return {
        "instance_count": m.get("design__instance__count"),
        "stdcell_count": m.get("design__instance__count__stdcell"),
        "area_um2": m.get("design__instance__area__stdcell") or m.get("design__instance__area"),
        "core_area_um2": m.get("design__core__area"),
        "die_area_um2": m.get("design__die__area"),
        "utilization": m.get("design__instance__utilization"),
        "power_w": m.get("power__total"),
        "power_internal_w": m.get("power__internal__total"),
        "power_switching_w": m.get("power__switching__total"),
        "power_leakage_w": m.get("power__leakage__total"),
        "setup_ws_ns": ws,
        "hold_ws_ns": pick_metric(m, "timing__hold__ws"),
        "setup_wns_ns": pick_metric(m, "timing__setup__wns"),
        "setup_vio": pick_metric(m, "timing__setup_vio__count"),
        "hold_vio": pick_metric(m, "timing__hold_vio__count"),
        "fmax_mhz_from_ws": fmax_mhz,
        "clock_period_ns": period,
        "drc": m.get("magic__drc_error__count"),
        "route_drc": m.get("route__drc_errors"),
        "lvs": m.get("design__lvs_error__count"),
        "flow_errors": m.get("flow__errors__count"),
        "metrics_source": source,
    }


def run_openlane(
    cfg: Path,
    tag: str,
    extra: list[str] | None = None,
    overwrite: bool = True,
) -> dict:
    env = os.environ.copy()
    env["PDK_ROOT"] = str(Path(os.environ.get("PDK_ROOT", Path.home() / ".volare")))
    period_ns = None
    try:
        period_ns = float(json.loads(cfg.read_text()).get("CLOCK_PERIOD") or CLOCK_NS)
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        period_ns = CLOCK_NS
    cmd = [
        str(ROOT / ".venv" / "bin" / "openlane"),
        "--docker-no-tty",
        "--dockerized",
        "--run-tag",
        tag,
    ]
    if overwrite:
        cmd.append("--overwrite")
    if extra:
        cmd.extend(extra)
    cmd.append(str(cfg))
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    run_dir = cfg.parent / "runs" / tag
    # OpenLane 2 writes next to the config by default? adder used openlane/runs from CWD.
    candidates = [
        ROOT / "openlane" / "runs" / tag,
        ROOT / "openlane" / "sweep" / "runs" / tag,
        ROOT / "runs" / tag,
        run_dir,
    ]
    found = next((p for p in candidates if (p / "final" / "metrics.json").exists()), None)
    if found is None:
        found = next(
            (p for p in candidates if (p / "50-openroad-stapostpnr" / "state_out.json").exists()),
            None,
        )
    out = {
        "returncode": proc.returncode,
        "tag": tag,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    if found:
        parsed = parse_openlane_metrics(found, period_ns=period_ns)
        parsed.pop("stdout_tail", None)
        out.update(parsed)
        out["run_dir"] = str(found)
        out.pop("stdout_tail", None)
        out.pop("stderr_tail", None)
    elif proc.returncode != 0:
        out["error"] = "openlane failed before metrics"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", default="4,8,16")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--pnr", action="store_true", help="also run OpenLane PnR")
    parser.add_argument("--pnr-max-n", type=int, default=8, help="skip PnR for tiles larger than this")
    args = parser.parse_args()
    tiles = [int(x) for x in args.tiles.split(",") if x]
    bits_l = [int(x) for x in args.bits.split(",") if x]

    results = []
    for n in tiles:
        for b in bits_l:
            print(f"=== gen {n}x{n}  {b}-bit ===", flush=True)
            rtl = gen_rtl(n, b)
            print(f"=== yosys {rtl.name} ===", flush=True)
            rpt = ROOT / "artifacts" / "eda" / f"yosys_{n}x{n}_b{b}.rpt"
            ys = yosys_synth(rtl, "ffn_tile_reg", rpt)
            ys.pop("stat_text_tail", None)
            row = {"n": n, "bits": b, "weights": n * n, "yosys": ys}
            if ys.get("area_um2"):
                row["area_um2_per_weight"] = ys["area_um2"] / (n * n)
                row["area_7nm_f2_um2"] = ys["area_um2"] * F2_SCALE_7NM
                row["area_7nm_sram_um2"] = ys["area_um2"] * SRAM_SCALE_7NM
            if args.pnr and n <= args.pnr_max_n:
                print(f"=== openlane {n}x{n} b{b} ===", flush=True)
                cfg = write_ol_config(n, b, rtl)
                tag = f"tile_{n}x{n}_b{b}"
                row["openlane"] = run_openlane(cfg, tag)
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "yosys"}, default=str), flush=True)

    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "clock_ns": CLOCK_NS,
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "note": (
            "Yosys+abc mapped area is pre-PnR standard-cell area. "
            "OpenLane numbers (when present) are post-route. "
            "7nm F² scale=(7/130)^2; SRAM scale=0.027/1.896 um²."
        ),
        "points": results,
    }
    out = ROOT / "artifacts" / "sky130_sweep.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
