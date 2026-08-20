"""Architecture eval for agents: gates, tok/s, area, energy, vs baseline.

Contract (stable keys): artifacts/eval.schema.json
Output:              artifacts/eval.json
Baseline:            artifacts/eval_baseline.json

  python scripts/eval_bench.py              # ingest existing artifacts (fast)
  python scripts/eval_bench.py --sim        # re-run DeltaNet equivalence
  python scripts/eval_bench.py --strict     # exit 2 on metric regression
  python scripts/eval_bench.py --promote    # copy eval.json → eval_baseline.json

Exit codes
  0  gates pass; no strict regression
  1  hard fail (equiv not PASS, or PnR DRC/LVS/setup vio)
  2  gates pass but --strict and a primary metric got worse

Primary metrics (used for direction)
  area_um2          lower is better   (OpenLane instance area, else Yosys)
  tok_s_per_pe      higher is better  (Fmax / cycles_per_token)
  nJ_per_token      lower is better   VECTORLESS — do not treat as real energy
  cycles_per_token  lower is better

Power is OpenSTA vectorless unless power_model says otherwise. Two RTL
styles can rank wrong on nJ/token until activity-annotated power exists.
Complexity (MACs vs seq) is the seq-length direction check; it needs no EDA.

Do not mix via-ROM reticle estimates with these stdcell numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.complexity import (  # noqa: E402
    complexity_table,
    crossover_seq,
    deltanet_macs_per_token,
)

SCHEMA = "qwen-ffn-rom.eval/v1"
OUT_PATH = ROOT / "artifacts" / "eval.json"
BASELINE_PATH = ROOT / "artifacts" / "eval_baseline.json"
SCHEMA_PATH = ROOT / "artifacts" / "eval.schema.json"

# Fraction worse that counts as a regression (PnR/Yosys noise).
REL_TOL = {
    "area_um2": 0.05,
    "tok_s_per_pe": 0.10,
    "nJ_per_token": 0.10,
    "cycles_per_token": 0.02,
}
HIGHER_IS_BETTER = {"tok_s_per_pe"}
PRIMARY = ("area_um2", "tok_s_per_pe", "nJ_per_token", "cycles_per_token")


def git_meta() -> dict:
    def run(args: list[str]) -> str | None:
        try:
            p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        except OSError:
            return None
        if p.returncode != 0:
            return None
        return p.stdout.strip() or None

    commit = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty)}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def activity_overlay(name: str, power: float | None, power_model: str | None) -> tuple:
    act = load_json(ROOT / "artifacts" / "activity_power.json")
    nJ = None
    if not act:
        return power, power_model, nJ
    for a in act.get("duts") or []:
        if a.get("name") == name and a.get("nJ_per_token") is not None:
            pw = (a.get("opensta") or {}).get("power_w")
            return (
                pw if pw is not None else power,
                a.get("power_model") or "opensta_vcd_activity_spef",
                a.get("nJ_per_token"),
            )
    return power, power_model, nJ


def derived(cycles: float | None, fmax_mhz: float | None, power_w: float | None, area_um2: float | None) -> dict:
    tok_s = None
    nJ = None
    tok_mm2 = None
    if cycles and fmax_mhz:
        tok_s = (fmax_mhz * 1e6) / cycles
    if tok_s and power_w is not None and power_w > 0:
        nJ = (power_w / tok_s) * 1e9
    if tok_s and area_um2:
        tok_mm2 = tok_s / (area_um2 / 1e6)
    return {
        "tok_s_per_pe": tok_s,
        "nJ_per_token": nJ,
        "tok_s_per_mm2": tok_mm2,
    }


LAYERS_08B = 24
LAYERS_27B = 64
FPGA_FMAX_MHZ = 200.0  # 7-series stand-in until Vivado Fmax


def attach_model(out: dict, cycles_layer, fmax_mhz, n_layers: int = LAYERS_08B) -> dict:
    """Headline is 0.8B model tok/s (one farm walks n_layers in time)."""
    cyc_m = (n_layers * cycles_layer) if cycles_layer else None
    tok_m = (fmax_mhz * 1e6 / cyc_m) if cyc_m and fmax_mhz else None
    cyc_27 = (LAYERS_27B * cycles_layer) if cycles_layer else None
    tok_27 = (fmax_mhz * 1e6 / cyc_27) if cyc_27 and fmax_mhz else None
    out["model_layers"] = n_layers
    out["cycles_per_layer"] = cycles_layer
    out["cycles_per_model_token"] = cyc_m
    out["model_tok_s"] = tok_m
    out["cycles_per_model_token_27b"] = cyc_27
    out["model_tok_s_27b"] = tok_27
    return out


def dut_delta() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "delta_equiv_gate.json")
    eda = load_json(ROOT / "artifacts" / "delta_d4_eda.json")
    if not equiv and not eda:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = (equiv or {}).get("cycles_per_token")
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    d = (equiv or {}).get("d") or (eda or {}).get("d") or 4
    out = {
        "name": "gated_delta_d4",
        "kind": "deltanet",
        "equiv_status": (equiv or {}).get("status"),
        "d": d,
        "mac": (equiv or {}).get("mac") or (eda or {}).get("mac") or 1,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": "opensta_vectorless" if power is not None else None,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "state_bytes": d * d * 2,
        "clock_period_ns": ol.get("clock_period_ns") or (eda or {}).get("clock_ns"),
    }
    out.update(derived(cycles, fmax, power, area))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


def dut_delta_d16() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "delta_d16_equiv.json")
    eda = load_json(ROOT / "artifacts" / "delta_d16_eda.json")
    if not equiv and not eda:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = (equiv or {}).get("cycles_per_token")
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    d = 16
    out = {
        "name": "gated_delta_d16",
        "kind": "deltanet",
        "top": "gated_delta_d16",
        "equiv_status": (equiv or {}).get("status"),
        "d": d,
        "mac": 1,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": "opensta_vectorless" if power is not None else None,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "state_bytes": d * d * 2,
        "clock_period_ns": ol.get("clock_period_ns") or (eda or {}).get("clock_ns") or 50.0,
        "note": "1-MAC FSM, not unrolled. tok_s_per_pe is one layer; model_tok_s walks 24 layers.",
    }
    out.update(derived(cycles, fmax, power, area))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


def dut_delta_dpar16() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "delta_dpar16_equiv.json")
    eda = load_json(ROOT / "artifacts" / "delta_dpar16_eda.json")
    if not equiv and not eda:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = (equiv or {}).get("cycles_per_token")
    fmax = ol.get("fmax_mhz_from_ws")
    fmax_src = "openlane" if fmax is not None else None
    if fmax is None:
        d4_eda = load_json(ROOT / "artifacts" / "delta_d4_eda.json") or {}
        fmax = (d4_eda.get("openlane") or {}).get("fmax_mhz_from_ws")
        fmax_src = "d4_openlane_proxy" if fmax is not None else None
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    d = (equiv or {}).get("d") or (eda or {}).get("d") or 16
    mac = (equiv or {}).get("mac") or (eda or {}).get("mac") or 16
    out = {
        "name": "gated_delta_d16_par",
        "kind": "deltanet",
        "top": "gated_delta_d16_par",
        "equiv_status": (equiv or {}).get("status"),
        "d": d,
        "mac": mac,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": "opensta_vectorless" if power is not None else None,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "state_bytes": d * d * 2,
        "clock_period_ns": ol.get("clock_period_ns") or (eda or {}).get("clock_ns") or 50.0,
        "fmax_source": fmax_src,
        "note": "D MACs: D-wide inner product; cycles ~O(D). Not unrolled D×D. PnR skipped (80k Yosys cells). tok_s_per_pe is one layer; model_tok_s walks 24 layers.",
    }
    out.update(derived(cycles, fmax, power, area))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


def dut_delta_fused(d: int) -> dict | None:
    equiv = load_json(ROOT / "artifacts" / f"delta_d{d}_fused_equiv.json")
    eda = load_json(ROOT / "artifacts" / f"delta_d{d}_fused_eda.json")
    if not equiv:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = equiv.get("cycles_per_token")
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    out = {
        "name": f"gated_delta_d{d}_fused",
        "kind": "deltanet_fused",
        "top": f"gated_delta_d{d}_fused",
        "equiv_status": equiv.get("status"),
        "d": d,
        "mac": equiv.get("mac") or d,
        "tokens_tested": equiv.get("tokens"),
        "cycles_per_token": cycles,
        "cycles_per_layer": equiv.get("cycles_per_layer"),
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "area_7nm_f2_um2": (eda or {}).get("area_7nm_f2_um2"),
        "yosys_cell_count": ys.get("cell_count"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "route_drc": ol.get("route_drc"),
        "note": (
            f"Column-stream fused PE, D={d}, D row-banks, 1-issue/cycle. "
            "cycles = D+2. 27B mixer uses D=128 = 130 clk, not the 524 four-sweep BRAM PE."
        ),
    }
    out.update(derived(cycles, fmax, ol.get("power_w"), area))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


def dut_delta_gqa() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "delta_gqa3_equiv.json")
    if not equiv:
        return None
    cycles = equiv.get("cycles_per_token")
    out = {
        "name": "gated_delta_gqa3_d16",
        "kind": "deltanet_gqa",
        "top": "gated_delta_gqa3_d16",
        "equiv_status": equiv.get("status"),
        "d": equiv.get("d"),
        "n_v": equiv.get("n_v"),
        "tokens_tested": equiv.get("tokens"),
        "cycles_per_token": cycles,
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    return out


def dut_rmsnorm_fast() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "rmsnorm_fast_equiv.json")
    if not equiv:
        return None
    cycles = equiv.get("cycles")
    out = {
        "name": "rmsnorm_fast_h8",
        "kind": "rmsnorm_nr",
        "top": "rmsnorm8",
        "equiv_status": equiv.get("status"),
        "h": equiv.get("h"),
        "n": equiv.get("n"),
        "cycles_per_token": cycles,
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    return out


def dut_ffn_swiglu() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "ffn_swiglu_equiv.json")
    if not equiv:
        return None
    cycles = equiv.get("cycles_per_tile")
    out = {
        "name": "ffn_tap_swiglu",
        "kind": "ffn_swiglu",
        "top": "ffn_tap_swiglu",
        "equiv_status": equiv.get("status"),
        "n": equiv.get("n"),
        "cycles_per_token": cycles,
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    return out


def dut_attn_online() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "attn_online_equiv.json")
    if not equiv:
        return None
    cycles = (equiv.get("measured") or [{}])[-1].get("cycles")
    out = {
        "name": "attn_online_d8_p4",
        "kind": "attn_online",
        "top": "attn_online_d8_p4",
        "equiv_status": equiv.get("status"),
        "d": equiv.get("d"),
        "p": equiv.get("p"),
        "cycles_per_token": cycles,
        "pipe": equiv.get("pipe"),
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    return out


def dut_attn_online_gqa() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "attn_online_gqa_equiv.json")
    if not equiv:
        return None
    cycles = (equiv.get("measured") or [{}])[-1].get("cycles")
    out = {
        "name": "attn_online_gqa_d8",
        "kind": "attn_online_gqa",
        "top": "attn_online_gqa_d8",
        "equiv_status": equiv.get("status"),
        "d": equiv.get("d"),
        "p": equiv.get("p"),
        "n_kv": equiv.get("n_kv"),
        "n_q_per_kv": equiv.get("n_q_per_kv"),
        "cycles_per_token": cycles,
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    return out


def dut_delta_bram(d: int) -> dict | None:
    equiv = load_json(ROOT / "artifacts" / f"delta_d{d}_bram_equiv.json")
    eda = load_json(ROOT / "artifacts" / f"gated_delta_d{d}_bram_fpga.json")
    if not equiv and not eda:
        return None
    ys = (eda or {}).get("yosys_xilinx") or {}
    cycles = (equiv or {}).get("cycles_per_layer") or (equiv or {}).get("cycles_per_token")
    fmax = (eda or {}).get("fmax_mhz") or FPGA_FMAX_MHZ
    fmax_src = (eda or {}).get("fmax_source") or "assumed_200mhz_xc7"
    out = {
        "name": f"gated_delta_d{d}_bram",
        "kind": "deltanet_fpga",
        "top": f"gated_delta_d{d}_bram",
        "platform": "fpga_xc7",
        "equiv_status": (equiv or {}).get("status"),
        "d": d,
        "mac": (equiv or {}).get("mac") or d,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": (equiv or {}).get("cycles_per_token"),
        "fmax_mhz": fmax,
        "fmax_source": fmax_src,
        "lut": ys.get("lut"),
        "ff": ys.get("ff"),
        "dsp": ys.get("dsp"),
        "bram18_equiv": ys.get("bram18_equiv"),
        "yosys_cell_count": ys.get("cell_count"),
        "note": (
            f"FPGA BRAM-S PE, D={d}, 1 DSP48/lane, overlapped rows (~4D cyc/layer). "
            "Headline model_tok_s is 0.8B (24 layers in time). Fmax assumed 200 MHz until Vivado."
        ),
    }
    out.update(derived(cycles, fmax, None, None))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


def dut_farm_d4() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "qwen08b_farm_d4_equiv.json")
    eda = load_json(ROOT / "artifacts" / "qwen08b_farm_d4_fpga.json")
    if not equiv and not eda:
        return None
    ys = (eda or {}).get("yosys_xilinx") or {}
    cycles_model = (equiv or {}).get("cycles_per_model_token") or (equiv or {}).get("cycles_per_token")
    cycles_layer = (equiv or {}).get("cycles_per_layer")
    fmax = (eda or {}).get("fmax_mhz") or FPGA_FMAX_MHZ
    fmax_src = (eda or {}).get("fmax_source") or "assumed_200mhz_xc7"
    out = {
        "name": "qwen08b_farm_d4",
        "kind": "model_farm_fpga",
        "top": "qwen08b_farm_d4",
        "platform": "fpga_xc7",
        "equiv_status": (equiv or {}).get("status"),
        "d": 4,
        "mac": 4,
        "n_layers": 24,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": cycles_model,
        "fmax_mhz": fmax,
        "fmax_source": fmax_src,
        "lut": ys.get("lut"),
        "ff": ys.get("ff"),
        "dsp": ys.get("dsp"),
        "bram18_equiv": ys.get("bram18_equiv"),
        "yosys_cell_count": ys.get("cell_count"),
        "note": (
            "0.8B mixer farm: one D=4 BRAM PE (1 DSP/lane, overlapped rows), "
            "24 layers in time, independent S. cycles_per_token is a model token. "
            "Fmax assumed 200 MHz until Vivado."
        ),
    }
    out.update(derived(cycles_model, fmax, None, None))
    out["tok_s_per_pe"] = (fmax * 1e6 / cycles_layer) if cycles_layer and fmax else None
    attach_model(out, cycles_layer, fmax, LAYERS_08B)
    out["cycles_per_model_token"] = cycles_model
    out["model_tok_s"] = (fmax * 1e6 / cycles_model) if cycles_model and fmax else None
    return out


def dut_heads16() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "qwen08b_heads16_d4_equiv.json")
    if not equiv:
        return None
    cycles_model = equiv.get("cycles_per_model_token") or equiv.get("cycles_per_token")
    fmax = FPGA_FMAX_MHZ
    out = {
        "name": "qwen08b_heads16_d4",
        "kind": "model_farm_fpga",
        "top": "qwen08b_heads16_d4",
        "platform": "fpga_xc7",
        "equiv_status": equiv.get("status"),
        "d": 4,
        "mac": 64,
        "n_heads": 16,
        "n_layers": 24,
        "tokens_tested": equiv.get("tokens"),
        "cycles_per_token": cycles_model,
        "fmax_mhz": fmax,
        "fmax_source": "assumed_200mhz_xc7",
        "note": (
            "16 independent D=4 farms (one DSP/lane each). Heads do not cut cycles. "
            "UART top in rtl/fpga_top.v; host scripts/fpga_host.py prints wall tok/s."
        ),
    }
    out.update(derived(cycles_model, fmax, None, None))
    attach_model(out, (cycles_model / LAYERS_08B) if cycles_model else None, fmax, LAYERS_08B)
    out["cycles_per_model_token"] = cycles_model
    out["model_tok_s"] = (fmax * 1e6 / cycles_model) if cycles_model and fmax else None
    return out


def dut_decoder_layer() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "decoder_layer_equiv.json")
    if not equiv:
        return None
    cycles = equiv.get("cycles_per_layer")
    out = {
        "name": "decoder_layer_d4_ffn8",
        "kind": "decoder_layer",
        "top": "decoder_layer",
        "equiv_status": equiv.get("status"),
        "d": 4,
        "hidden": 8,
        "tokens_tested": equiv.get("tokens"),
        "cycles_per_token": cycles,
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, None, None, None))
    attach_model(out, cycles, None, LAYERS_08B)
    return out


def dut_qwen_layer() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "qwen_layer_equiv.json")
    eda = load_json(ROOT / "artifacts" / "qwen_layer_eda.json")
    if not equiv:
        return None
    cycles = equiv.get("cycles_delta_layer")
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    area = ol.get("area_um2") or ys.get("area_um2")
    fmax = (eda or {}).get("fmax_mhz_sky130") or ol.get("fmax_mhz_from_ws")
    area_src = (eda or {}).get("area_source") or ("openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None))
    out = {
        "name": "qwen_layer_h16",
        "kind": "qwen_layer",
        "top": "qwen_layer",
        "equiv_status": equiv.get("status"),
        "d": 4,
        "hidden": 16,
        "heads": 4,
        "tokens_tested": equiv.get("tokens"),
        "cycles_per_token": cycles,
        "cycles_attn_layer_mean": equiv.get("cycles_attn_layer_mean"),
        "area_um2": area,
        "area_source": area_src,
        "area_7nm_f2_um2": (eda or {}).get("area_7nm_f2_um2"),
        "fmax_mhz": fmax,
        "fmax_source": (eda or {}).get("fmax_source") or ("openlane_postroute_ws" if fmax else None),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "route_drc": ol.get("route_drc"),
        "magic_drc_lvs": "unmeasured",
        "note": equiv.get("note"),
    }
    out.update(derived(cycles, fmax, ol.get("power_w"), area))
    return out


def dut_hybrid() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "hybrid_layer_equiv.json")
    eda = load_json(ROOT / "artifacts" / "hybrid_layer_eda.json")
    if not equiv and not eda:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = (equiv or {}).get("cycles_per_token")
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    out = {
        "name": "hybrid_layer_d4_ffn8",
        "kind": "hybrid_layer",
        "top": "hybrid_layer_stub",
        "equiv_status": (equiv or {}).get("status"),
        "d": (equiv or {}).get("d") or 4,
        "n": (equiv or {}).get("n") or 8,
        "bits": 4,
        "tokens_tested": (equiv or {}).get("tokens"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": "opensta_vectorless" if power is not None else None,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "clock_period_ns": ol.get("clock_period_ns") or (eda or {}).get("clock_ns"),
        "schedule": (equiv or {}).get("schedule"),
        "note": "layer tok/s (mixer+FFN on shared x), not FFN per-PE. model_tok_s walks 24 layers.",
    }
    out.update(derived(cycles, fmax, power, area))
    attach_model(out, cycles, fmax, LAYERS_08B)
    return out


XBAR_SPECS = (
    ("ffn_col_serial_8x8_b4", "serial_csd", "ffn_col_serial"),
    ("ffn_rom_tap_8x8_b4", "rom_tap", "ffn_rom_tap_reg"),
    ("ffn_rom_fetch_8x8_b4", "rom_fetch", "ffn_rom_fetch"),
)


def dut_xbar(name: str, kind: str, top: str) -> dict | None:
    equiv = load_json(ROOT / "artifacts" / f"{name}_equiv.json")
    eda_all = load_json(ROOT / "artifacts" / "xbar_eda.json")
    eda = None
    if eda_all:
        for d in eda_all.get("duts") or []:
            if d.get("name") == name:
                eda = d
                break
    if not equiv and not eda:
        return None
    ol = (eda or {}).get("openlane") or {}
    ys = (eda or {}).get("yosys") or {}
    cycles = (equiv or {}).get("cycles_per_token")
    if cycles is None and kind == "rom_tap":
        cycles = 1.0
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    power_model = "opensta_vectorless" if power is not None else None
    power, power_model, nJ_act = activity_overlay(name, power, power_model)
    out = {
        "name": name,
        "kind": kind,
        "top": top,
        "equiv_status": (equiv or {}).get("status"),
        "n": 8,
        "bits": 4,
        "vectors_tested": (equiv or {}).get("vectors"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": power_model,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "clock_period_ns": ol.get("clock_period_ns") or (eda_all or {}).get("clock_ns"),
        "note": "stdcell stand-in; do not treat rom_* area as via-ROM density",
    }
    out.update(derived(cycles, fmax, power, area))
    if nJ_act is not None:
        out["nJ_per_token"] = nJ_act
    return out


def dut_ffn_8x8() -> dict | None:
    equiv = load_json(ROOT / "artifacts" / "equiv_gate.json")
    sweep = load_json(ROOT / "artifacts" / "sky130_sweep.json")
    point = None
    if sweep:
        for p in sweep.get("points") or []:
            if p.get("n") == 8 and p.get("bits") == 4:
                point = p
                break
    if not equiv and not point:
        return None
    ol = (point or {}).get("openlane") or {}
    ys = (point or {}).get("yosys") or {}
    cycles = 1.0
    fmax = ol.get("fmax_mhz_from_ws")
    area = ol.get("area_um2") or ys.get("area_um2")
    area_src = "openlane" if ol.get("area_um2") else ("yosys" if ys.get("area_um2") else None)
    power = ol.get("power_w")
    power_model = "opensta_vectorless" if power is not None else None
    power, power_model, nJ_act = activity_overlay("ffn_tile_8x8_b4", power, power_model)
    out = {
        "name": "ffn_tile_8x8_b4",
        "kind": "ffn",
        "equiv_status": (equiv or {}).get("status"),
        "n": 8,
        "bits": 4,
        "vectors_tested": (equiv or {}).get("vectors"),
        "cycles_per_token": cycles,
        "fmax_mhz": fmax,
        "area_um2": area,
        "area_source": area_src,
        "yosys_cell_count": ys.get("cell_count"),
        "openlane_instances": ol.get("instance_count"),
        "power_w": power,
        "power_model": power_model,
        "drc": ol.get("drc"),
        "lvs": ol.get("lvs"),
        "setup_vio": ol.get("setup_vio"),
        "hold_vio": ol.get("hold_vio"),
        "clock_period_ns": ol.get("clock_period_ns"),
    }
    out.update(derived(cycles, fmax, power, area))
    if nJ_act is not None:
        out["nJ_per_token"] = nJ_act
    return out


def complexity_block() -> dict:
    d_pe = 4
    d_head = 128
    return {
        "note": (
            "Decode MACs for one new token. Softmax grows with seq; DeltaNet does not. "
            "d=4 is the signed-off PE; d=128 is a typical head size for scaling talk."
        ),
        "pe": {
            "d": d_pe,
            "delta_macs_per_token": deltanet_macs_per_token(d_pe),
            "crossover_seq": crossover_seq(d_pe),
            "table": complexity_table(d_pe),
        },
        "head_assumed": {
            "d": d_head,
            "delta_macs_per_token": deltanet_macs_per_token(d_head),
            "crossover_seq": crossover_seq(d_head),
            "table": complexity_table(d_head, (256, 1024, 4096, 32768)),
        },
    }


def collect_gates(duts: list[dict]) -> dict:
    notes = []
    equiv_ok = True
    signoff_ok = True
    for d in duts:
        st = d.get("equiv_status")
        if st is not None and st != "PASS":
            equiv_ok = False
        if d.get("equiv_status") is None:
            notes.append(f"{d['name']}: no equiv json")
        for k in ("drc", "lvs", "setup_vio"):
            v = d.get(k)
            if v is None:
                continue
            if v not in (0, 0.0):
                signoff_ok = False
        if d.get("power_model") == "opensta_vectorless":
            notes.append(f"{d['name']}: nJ/token is vectorless, not activity-annotated")
        elif d.get("power_model") == "opensta_vcd_activity_spef":
            notes.append(f"{d['name']}: nJ/token is VCD-activity OpenSTA on post-route SPEF")
    missing_equiv = [d["name"] for d in duts if d.get("equiv_status") != "PASS"]
    return {
        "equiv_ok": equiv_ok and all(d.get("equiv_status") == "PASS" for d in duts),
        "signoff_ok": signoff_ok,
        "all_pass": equiv_ok and signoff_ok and not missing_equiv,
        "notes": notes,
    }


def worse(metric: str, new: float, old: float) -> bool:
    if old == 0:
        return new != 0 and metric not in HIGHER_IS_BETTER
    rel = (new - old) / abs(old)
    tol = REL_TOL.get(metric, 0.05)
    if metric in HIGHER_IS_BETTER:
        return rel < -tol
    return rel > tol


def better(metric: str, new: float, old: float) -> bool:
    if old == 0:
        return False
    rel = (new - old) / abs(old)
    tol = REL_TOL.get(metric, 0.05)
    if metric in HIGHER_IS_BETTER:
        return rel > tol
    return rel < -tol


def compare(current: dict, baseline: dict) -> dict:
    improvements = []
    regressions = []
    mixed_duts = []
    base_duts = {d["name"]: d for d in baseline.get("duts") or []}
    for dut in current.get("duts") or []:
        prev = base_duts.get(dut["name"])
        if not prev:
            continue
        imp = []
        reg = []
        for m in PRIMARY:
            a, b = dut.get(m), prev.get(m)
            if a is None or b is None:
                continue
            if m == "nJ_per_token" and dut.get("power_model") != prev.get("power_model"):
                continue
            rec = {
                "dut": dut["name"],
                "metric": m,
                "before": b,
                "after": a,
                "rel": (a - b) / b if b else None,
            }
            if worse(m, a, b):
                reg.append(rec)
                regressions.append(rec)
            elif better(m, a, b):
                imp.append(rec)
                improvements.append(rec)
        if imp and reg:
            mixed_duts.append(dut["name"])
    if not base_duts:
        direction = "no_baseline"
    elif regressions and not improvements:
        direction = "worse"
    elif improvements and not regressions:
        direction = "better"
    elif regressions and improvements:
        direction = "mixed"
    else:
        direction = "unchanged"
    return {
        "direction": direction,
        "improvements": improvements,
        "regressions": regressions,
        "mixed_duts": mixed_duts,
        "baseline_commit": (baseline.get("git") or {}).get("commit"),
        "rel_tol": REL_TOL,
    }


def build_report() -> dict:
    duts = [d for d in (dut_delta(), dut_delta_d16(), dut_delta_dpar16(), dut_ffn_8x8(), dut_hybrid()) if d]
    for d in (16, 128):
        row = dut_delta_fused(d)
        if row:
            duts.append(row)
    gqa = dut_delta_gqa()
    if gqa:
        duts.append(gqa)
    for row in (dut_attn_online(), dut_attn_online_gqa(), dut_rmsnorm_fast(), dut_ffn_swiglu()):
        if row:
            duts.append(row)
    for d in (4, 16, 128):
        row = dut_delta_bram(d)
        if row:
            duts.append(row)
    farm = dut_farm_d4()
    if farm:
        duts.append(farm)
    heads = dut_heads16()
    if heads:
        duts.append(heads)
    dec = dut_decoder_layer()
    if dec:
        duts.append(dec)
    qlayer = dut_qwen_layer()
    if qlayer:
        duts.append(qlayer)
    for name, kind, top in XBAR_SPECS:
        row = dut_xbar(name, kind, top)
        if row:
            duts.append(row)
    sweep = load_json(ROOT / "artifacts" / "xbar_n_sweep.json")
    gates = collect_gates(duts)
    return {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_meta(),
        "gates": gates,
        "duts": duts,
        "n_sweep": sweep,
        "complexity": complexity_block(),
        "caveats": [
            "Headline tok/s is model_tok_s: one farm walks all layers in time (24 for 0.8B, 64 for Qwen3.8-27B).",
            "tok_s_per_pe is still one layer / one PE (kept for baseline compare). Do not quote it as a model rate.",
            "FPGA Fmax is assumed 200 MHz (Artix-7 class) until Vivado; labeled fmax_source=assumed_200mhz_xc7.",
            "nJ_per_token uses OpenSTA VCD-activity on post-route SPEF when artifacts/activity_power.json exists; else vectorless.",
            "Do not mix these stdcell areas with via-ROM reticle estimates.",
            "FFN 8x8 vs DeltaNet D=4 are different jobs; compare same dut.name across runs.",
            "rom_tap/rom_fetch/col_serial share W with ffn_tile_8x8_b4; rank those four on area/cycles/Fmax.",
            "hybrid_layer_d4_ffn8 is one decode layer (D=4 mixer + 8x8 FFN); tok/s is the layer, not 75 M FFN per-PE.",
            "gated_delta_d16 is the same 1-MAC FSM as d4; cycles scale ~D^2. Do not unroll.",
            "gated_delta_d16_par is D MACs (inner product / row); cycles ~O(D). rom_tap/rom_fetch are stdcell, not via-ROM GDS.",
            "gated_delta_d128_fused is the 27B mixer: 130 clk/layer (D+2), bit-exact. "
            "gated_delta_gqa3 runs 3 V-heads at that same D+2. The 524-clk BRAM PE is four unfused sweeps.",
            "attn_online is fused online-softmax, P-way int4 KV: cycles ceil(S/P)+2. "
            "GQA 4×6 matches 1-head cycles. 27B uses P=512 → 16 layers ≤ 1056 clk at 32k.",
            "rmsnorm_fast is Newton rsqrt, 2 clk (was 58 restoring). SiLU/gate is folded "
            "into ffn_tap_swiglu at the same 2 clk handshake — 0 extra token clocks. "
            "27B norms+activation = 256 clk/token.",
        ],
    }


def write_schema() -> None:
    SCHEMA_PATH.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "primary_metrics": list(PRIMARY),
                "higher_is_better": sorted(HIGHER_IS_BETTER),
                "rel_tol": REL_TOL,
                "exit_codes": {
                    "0": "gates pass; no strict regression",
                    "1": "equiv fail or PnR DRC/LVS/setup vio",
                    "2": "--strict and a primary metric regressed",
                },
            },
            indent=2,
        )
        + "\n"
    )


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1e5:
            return f"{v:.3e}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4g}"
    return str(v)


def print_scorecard(rep: dict, verdict: dict) -> None:
    print(f"eval {rep['schema']}  {rep['utc']}  git={rep['git'].get('commit')} dirty={rep['git'].get('dirty')}")
    g = rep["gates"]
    print(f"gates  equiv_ok={g['equiv_ok']}  signoff_ok={g['signoff_ok']}  all_pass={g['all_pass']}")
    print(f"{'dut':26} {'equiv':5} {'cycL':>5} {'cycM':>6} {'MHz':>7} {'model tok/s':>12} {'µm²':>10} {'nJ/tok':>10}")
    for d in rep["duts"]:
        print(
            f"{d['name']:26} {str(d.get('equiv_status') or '?'):5} "
            f"{fmt(d.get('cycles_per_layer') or d.get('cycles_per_token')):>5} "
            f"{fmt(d.get('cycles_per_model_token')):>6} {fmt(d.get('fmax_mhz')):>7} "
            f"{fmt(d.get('model_tok_s')):>12} {fmt(d.get('area_um2')):>10} "
            f"{fmt(d.get('nJ_per_token')):>10}"
        )
    cx = rep["complexity"]["head_assumed"]
    print(
        f"complexity  d={cx['d']}  delta_macs/tok={cx['delta_macs_per_token']}  "
        f"attn overtakes at seq≈{cx['crossover_seq']:.0f}"
    )
    print(f"direction   {verdict['direction']}")
    for rec in verdict.get("improvements") or []:
        print(f"  + {rec['dut']} {rec['metric']}: {fmt(rec['before'])} → {fmt(rec['after'])}")
    for rec in verdict.get("regressions") or []:
        print(f"  - {rec['dut']} {rec['metric']}: {fmt(rec['before'])} → {fmt(rec['after'])}")
    for n in g.get("notes") or []:
        print(f"note  {n}")
    sweep = rep.get("n_sweep") or {}
    pts = sweep.get("points") or []
    if pts:
        print(
            f"{'n':>4} {'style':14} {'abc_ns':>8} {'pred_MHz':>9} {'pnr_MHz':>8} "
            f"{'dies50':>6} {'tok/s':>10}"
        )
        for p in pts:
            st = p.get("sta") or {}
            ol = p.get("openlane") or {}
            print(
                f"{p.get('n'):4} {str(p.get('style')):14} "
                f"{fmt(st.get('abc_delay_ns')):>8} {fmt(st.get('pred_pnr_fmax_mhz')):>9} "
                f"{fmt(ol.get('fmax_mhz_from_ws')):>8} "
                f"{str(st.get('sta_dies_50ns')):>6} {fmt(p.get('tok_s_pnr') or p.get('tok_s_pred')):>10}"
            )


def run_sim() -> None:
    jobs = [
        (["make", "-C", str(ROOT / "tb"), "-f", "Makefile.delta"], "sim-delta"),
        (["make", "-C", str(ROOT / "tb"), "-f", "Makefile.xbar", "XBAR_DUT=ffn_col_serial"], "sim-xbar-serial"),
        (["make", "-C", str(ROOT / "tb"), "-f", "Makefile.xbar", "XBAR_DUT=ffn_rom_tap"], "sim-xbar-tap"),
        (["make", "-C", str(ROOT / "tb"), "-f", "Makefile.hybrid"], "sim-hybrid"),
    ]
    for cmd, label in jobs:
        print(f"=== {label} ===", flush=True)
        p = subprocess.run(cmd, cwd=ROOT)
        if p.returncode != 0:
            raise SystemExit(f"equivalence sim failed ({label})")


def promote(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text())
    print(f"promoted {src} → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="re-run DeltaNet cocotb gate before ingest")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--strict", action="store_true", help="exit 2 if a primary metric regressed")
    parser.add_argument("--promote", action="store_true", help="write this eval as the new baseline")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        assert deltanet_macs_per_token(4) == 4 * 16 + 4
        assert crossover_seq(128) > 100
        t = complexity_table(4, (64, 4096))
        assert t[0]["delta_macs"] == t[1]["delta_macs"]
        assert t[1]["attn_decode_macs"] > t[0]["attn_decode_macs"]
        print("self-check ok")
        return

    if args.sim:
        run_sim()

    write_schema()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    rep = build_report()
    if not rep["duts"]:
        print("no DUT artifacts found; run make sim-delta / make equiv / make eda-delta", file=sys.stderr)
        raise SystemExit(1)

    baseline = load_json(args.baseline) if args.baseline.exists() else None
    verdict = compare(rep, baseline) if baseline else {"direction": "no_baseline", "improvements": [], "regressions": []}
    rep["verdict"] = verdict
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print_scorecard(rep, verdict)
    print(f"wrote {args.out}")

    if args.promote:
        promote(args.out, args.baseline)

    if not rep["gates"]["all_pass"]:
        raise SystemExit(1)
    if args.strict and verdict.get("regressions"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
