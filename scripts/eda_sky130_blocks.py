#!/usr/bin/env python3
"""Sky130 OpenLane Fmax of the distinct 27B blocks (one instance each).

Chip Fmax = min(block Fmax). tok/s = Fmax / 6786 at 4k farm-hidden.
Die area is ROM-compiler density × 27B × bit-width — not a full-model PnR.

  python scripts/eda_sky130_blocks.py --pnr
  python scripts/eda_sky130_blocks.py --pnr --only mixer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rom_estimator import ARRAY_EFFICIENCY, rom_cell_um2  # noqa: E402
from sky130_sweep import (  # noqa: E402
    F2_SCALE_7NM,
    SRAM_SCALE_7NM,
    parse_openlane_metrics,
    run_openlane,
    yosys_synth,
)

# FO4 130 nm hd (~50 ps) → 7 nm (~8 ps) ≈ 6.3×. Not 130/7 = 18.6× (area/F²).
FMAX_SCALE_7NM = 6.5
CLK_PER_TOKEN_4K = 6786
# Compiler-macro mixer/attn/rsqrt: not the flatten-to-flops STA (3.23 MHz).
# Anchors: D=4 mixer 41.3 MHz and H=16 qwen_layer 34.8 MHz post-route.
# Band is ~2×; do not treat as 99% STA.
EST_SKY130_MHZ_LO = 20.0
EST_SKY130_MHZ_HI = 40.0
EST_SKY130_MHZ_NOM = 30.0
EST_NOTE = (
    "Estimate, not post-route. D=128 mixer STA (3.23 MHz) is a fanout-1922 "
    "flop-RAM artifact; do not use. Band is D=4 mixer 41 MHz and H=16 layer "
    "35 MHz, with a 128-tree EX a bit slower. Attn sweep / rsqrt unrouted, "
    "same family. ±2×. 7 nm is ×6.5 FO4, labeled projected."
)
P_27B = 27e9
WIRE_FACTOR = 4.37  # OpenLane path / ABC, calibrated on signed-off 8×8 tile

BLOCKS = [
    {
        "id": "mixer",
        "top": "gated_delta_d128_sta",
        "cfg": "openlane/gated_delta_d128_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "sta" / "delta_s_col_ram.v",
            ROOT / "rtl" / "add_tree_bal.v",
            ROOT / "rtl" / "gated_delta_fused.v",
            ROOT / "rtl" / "sta" / "gated_delta_d128_sta.v",
        ],
        "note": (
            "One D=128 fused mixer PE (130 clk = D+2). SRAM is a 1-cycle "
            "compiler-macro stub; OpenLane closes the combo 4-mul EX, not 128×128 bits."
        ),
    },
    {
        "id": "attn",
        "top": "attn_sweep_sta",
        "cfg": "openlane/attn_sweep_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "sta" / "kv_seq_ram.v",
            ROOT / "rtl" / "add_tree_bal.v",
            ROOT / "rtl" / "attn_exp_lut.v",
            ROOT / "rtl" / "attn_sweep_pe.v",
            ROOT / "rtl" / "sta" / "attn_sweep_sta.v",
        ],
        "note": (
            "One streaming-sweep unit, D=256, 4 KV-SRAM banks (1R1W stub). "
            "Generate D-dot + tree, not a 256-ripple loop. Chip P_eff=512 is "
            "N copies of this path plus a log-P reduce."
        ),
    },
    {
        "id": "rsqrt",
        "top": "rmsnorm_rsqrt_sta",
        "cfg": "openlane/rmsnorm_rsqrt_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "add_tree_bal.v",
            ROOT / "rtl" / "rsqrt_lut.v",
            ROOT / "rtl" / "inv_rsqrt.v",
            ROOT / "rtl" / "rmsnorm.v",
            ROOT / "rtl" / "rmsnorm128.v",
            ROOT / "rtl" / "sta" / "rmsnorm_rsqrt_sta.v",
        ],
        "note": "Newton rsqrt + RMSNorm handshake, H=128 slice (mixer width). Combo LUT+1 NR.",
    },
    {
        "id": "ffn_tap",
        "top": "ffn_tap_adder_sta",
        "cfg": "openlane/ffn_tap_adder_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "add_tree_bal.v",
            ROOT / "rtl" / "ffn_rom_tap.v",
            ROOT / "rtl" / "sta" / "ffn_tap_adder_sta.v",
        ],
        "note": "One 8×8 via-tap plus a 128-input farm adder-tree slice.",
    },
    {
        "id": "silu",
        "top": "silu_gate_sta",
        "cfg": "openlane/silu_gate_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "silu_lut.v",
            ROOT / "rtl" / "ffn_rom_tap.v",
            ROOT / "rtl" / "ffn_tap_swiglu.v",
            ROOT / "rtl" / "sta" / "silu_gate_sta.v",
        ],
        "note": "SiLU LUT + SwiGLU gate in the tap handshake (0 extra token clocks).",
    },
    {
        "id": "sequencer",
        "top": "layer_sequencer_sta",
        "cfg": "openlane/layer_sequencer_sta.json",
        "abc_fast": True,
        "rtl": [
            ROOT / "rtl" / "layer_sequencer.v",
            ROOT / "rtl" / "sta" / "layer_sequencer_sta.v",
        ],
        "note": "Layer handshake FSM (rms → mixer|attn → rms → ffn). PE dones stubbed.",
    },
]

SKIP_STA = ["--skip", "OpenROAD.STAPrePNR", "--skip", "OpenROAD.STAMidPNR"]
LARGE_BLOCKS = {"mixer", "attn", "rsqrt"}


def parse_sta_log(path: Path, period_ns: float) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"^wns\s+(-?\d+\.?\d*)", text, re.M)
    if not m:
        return None
    wns = float(m.group(1))
    path_ns = period_ns - wns
    fmax = (1000.0 / path_ns) if path_ns > 0 else None
    start = None
    sm = re.search(r"Startpoint:\s+(\S+)", text)
    end = None
    em = re.search(r"Endpoint:\s+(\S+)", text)
    if sm:
        start = sm.group(1)
    if em:
        end = em.group(1)
    return {
        "wns_ns": wns,
        "period_ns": period_ns,
        "path_ns": path_ns,
        "fmax_mhz": fmax,
        "startpoint": start,
        "endpoint": end,
        "log": str(path.relative_to(ROOT)),
    }


def mixer_sta() -> dict | None:
    cts = parse_sta_log(ROOT / "artifacts" / "eda" / "sta_mixer_d128_cts_ss.log", 100.0)
    pre = parse_sta_log(ROOT / "artifacts" / "eda" / "sta_mixer_d128_ss.log", 100.0)
    if not cts and not pre:
        return None
    out: dict = {
        "corner": "sky130_fd_sc_hd ss 100C 1.60V",
        "note": (
            "RepairDesignPostGPL OOM in 8 GiB Docker VM; TritonRoute not run. "
            "Chip Fmax uses post-CTS placement-RC OpenSTA, not ABC &nf."
        ),
    }
    if cts:
        out.update(cts)
        out["stage"] = "post_cts_placement_rc"
    if pre:
        out["cell_only_preroute"] = {**pre, "stage": "synth_netlist_ideal_wire"}
        if not cts:
            out.update(pre)
            out["stage"] = "synth_netlist_ideal_wire"
    return out


def load_existing_blocks() -> dict[str, dict]:
    p = ROOT / "artifacts" / "sky130_27b_blocks.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {b["id"]: b for b in data.get("blocks") or [] if isinstance(b, dict) and "id" in b}


def rom_die(nm: float, sram_um2: float, params: float, bits: int) -> dict:
    cell = rom_cell_um2(nm, sram_um2)
    um2 = params * bits * cell / ARRAY_EFFICIENCY
    return {
        "rom_cell_um2": cell,
        "bitwidth": bits,
        "params": params,
        "array_efficiency": ARRAY_EFFICIENCY,
        "mm2": um2 / 1e6,
        "formula": "params × bitwidth × rom_cell / array_efficiency",
    }


def ingest_existing(cfg: Path, tag: str) -> dict | None:
    run = ROOT / "openlane" / "runs" / tag
    if not (run / "final" / "metrics.json").exists() and not (
        run / "50-openroad-stapostpnr" / "state_out.json"
    ).exists():
        return None
    period = float(json.loads(cfg.read_text()).get("CLOCK_PERIOD") or 50.0)
    parsed = parse_openlane_metrics(run, period_ns=period)
    parsed["tag"] = tag
    parsed["run_dir"] = str(run)
    parsed["ingested"] = True
    return parsed


def run_block(block: dict, *, pnr: bool, ingest: bool) -> dict:
    top = block["top"]
    tag = top
    cfg = ROOT / block["cfg"]
    rec: dict = {
        "id": block["id"],
        "top": top,
        "note": block["note"],
    }
    if ingest:
        ol = ingest_existing(cfg, tag)
        prev = load_existing_blocks().get(block["id"]) or {}
        if ol:
            rec = {**prev, **rec, "openlane": ol}
            print(f"  ingested {tag} Fmax={ol.get('fmax_mhz_from_ws')}", flush=True)
            if block["id"] == "mixer":
                sta = mixer_sta()
                if sta:
                    rec["sta"] = sta
            return rec
        if not pnr and prev:
            rec = {**prev, **rec}
            if block["id"] == "mixer":
                sta = mixer_sta()
                if sta:
                    rec["sta"] = sta
            print(f"  kept existing {tag} (no post-route metrics)", flush=True)
            return rec
    rpt = ROOT / "artifacts" / "eda" / f"yosys_{tag}.rpt"
    print(f"=== yosys {top} ===", flush=True)
    ys = yosys_synth(block["rtl"], top, rpt, abc_fast=bool(block["abc_fast"]))
    ys.pop("stat_text_tail", None)
    abc_ns = ys.get("abc_delay_ns")
    pred = (abc_ns * WIRE_FACTOR) if abc_ns else None
    rec["yosys"] = ys
    rec["pred_pnr_fmax_mhz"] = (1000.0 / pred) if pred and pred > 0 else None
    rec["pred_pnr_note"] = (
        f"ABC delay × {WIRE_FACTOR} wire factor (calibrated on 8×8 OpenLane/ABC). "
        "Not post-route."
        if pred
        else None
    )
    if ys.get("area_um2"):
        rec["area_7nm_f2_um2"] = ys["area_um2"] * F2_SCALE_7NM
        rec["area_7nm_sram_um2"] = ys["area_um2"] * SRAM_SCALE_7NM
    if pnr:
        run = ROOT / "openlane" / "runs" / tag
        cts_odb = run / "30-openroad-cts" / f"{top}.odb"
        if block["id"] == "mixer" and cts_odb.exists():
            print(
                f"=== skip openlane {tag} (CTS odb present; do not --overwrite) ===",
                flush=True,
            )
        else:
            extra = SKIP_STA if block["id"] in LARGE_BLOCKS else None
            print(f"=== openlane {tag} ===", flush=True)
            rec["openlane"] = run_openlane(cfg, tag, extra=extra)
            print(
                f"  Fmax={rec['openlane'].get('fmax_mhz_from_ws')} "
                f"ws={rec['openlane'].get('setup_ws_ns')} "
                f"rc={rec['openlane'].get('returncode')}",
                flush=True,
            )
    if block["id"] == "mixer":
        sta = mixer_sta()
        if sta:
            rec["sta"] = sta
    return rec


def block_fmax(b: dict) -> tuple[str, float, str] | None:
    """Post-route WS only. Mixer flop-RAM STA and ABC &nf are not chip Fmax."""
    ol = b.get("openlane") or {}
    src = ol.get("metrics_source") or ""
    fm = ol.get("fmax_mhz_from_ws")
    if isinstance(fm, (int, float)) and (
        "stapostpnr" in src or src.endswith("metrics.json") or "final" in src
    ):
        return (b["id"], float(fm), "openlane_postroute_ws")
    if isinstance(fm, (int, float)) and not ol.get("error"):
        return (b["id"], float(fm), "openlane_ws")
    return None


def tok_at(mhz: float) -> float:
    return mhz * 1e6 / CLK_PER_TOKEN_4K


def summarize(blocks: list[dict]) -> dict:
    fmaxes = [t for t in (block_fmax(b) for b in blocks) if t]
    closed_id, closed_min, closed_src = (
        min(fmaxes, key=lambda t: t[1]) if fmaxes else (None, None, None)
    )
    mix_sta = None
    for b in blocks:
        if b.get("id") == "mixer" and isinstance((b.get("sta") or {}).get("fmax_mhz"), (int, float)):
            mix_sta = float(b["sta"]["fmax_mhz"])
            break
    lo, hi, nom = EST_SKY130_MHZ_LO, EST_SKY130_MHZ_HI, EST_SKY130_MHZ_NOM
    return {
        "clk_per_token_4k_farm_hidden": CLK_PER_TOKEN_4K,
        "sky130_chip_fmax_mhz": nom,
        "sky130_chip_fmax_block": "mixer_attn_rsqrt_family",
        "sky130_fmax_source": "estimate_compiler_macro_20_40",
        "sky130_tok_s_4k": tok_at(nom),
        "fmax_scale_sky130_to_7nm": FMAX_SCALE_7NM,
        "fmax_7nm_mhz_projected": nom * FMAX_SCALE_7NM,
        "tok_s_7nm_4k_projected": tok_at(nom * FMAX_SCALE_7NM),
        "projection_note": (
            "Headline Fmax is an estimate (20–40 MHz Sky130), not mixer STA. "
            "7 nm = that estimate × 6.5 FO4. Not 7 nm STA, not (7/130)²."
        ),
        "block_fmax_mhz": {b[0]: {"mhz": b[1], "source": b[2]} for b in fmaxes},
        "measured_closed_min_mhz": closed_min,
        "measured_closed_block": closed_id,
        "measured_closed_source": closed_src,
        "do_not_use_mixer_sta_mhz": mix_sta,
        "estimate": {
            "kind": "estimate",
            "not_99_percent": True,
            "uncertainty": "about 2×",
            "sky130_mhz_lo": lo,
            "sky130_mhz_nom": nom,
            "sky130_mhz_hi": hi,
            "tok_s_4k_lo": tok_at(lo),
            "tok_s_4k_nom": tok_at(nom),
            "tok_s_4k_hi": tok_at(hi),
            "fmax_7nm_mhz_lo": lo * FMAX_SCALE_7NM,
            "fmax_7nm_mhz_nom": nom * FMAX_SCALE_7NM,
            "fmax_7nm_mhz_hi": hi * FMAX_SCALE_7NM,
            "tok_s_7nm_4k_lo": tok_at(lo * FMAX_SCALE_7NM),
            "tok_s_7nm_4k_nom": tok_at(nom * FMAX_SCALE_7NM),
            "tok_s_7nm_4k_hi": tok_at(hi * FMAX_SCALE_7NM),
            "blocks": {
                "mixer": {"mhz_lo": lo, "mhz_hi": hi, "mhz_nom": nom},
                "attn": {"mhz_lo": lo, "mhz_hi": hi, "mhz_nom": nom},
                "rsqrt": {"mhz_lo": lo, "mhz_hi": hi, "mhz_nom": nom},
            },
            "anchors_mhz": {
                "gated_delta_d4_postroute": 41.28,
                "qwen_layer_h16_postroute": 34.81,
                "ffn_rom_tap_8x8_postroute": 75.56,
                "ffn_tap_adder_postroute": 72.64,
                "silu_postroute": 65.19,
                "sequencer_postroute": 85.50,
            },
            "note": EST_NOTE,
        },
    }


def rom_die_18f2(nm: float, params: float, bits: int) -> dict:
    """Uncapped via-ROM compiler density: 18 F². Not the SRAM/4 FinFET floor."""
    f_um = nm / 1000.0
    cell = 18.0 * f_um * f_um
    um2 = params * bits * cell / ARRAY_EFFICIENCY
    return {
        "rom_cell_um2": cell,
        "bitwidth": bits,
        "params": params,
        "array_efficiency": ARRAY_EFFICIENCY,
        "f2": 18,
        "mm2": um2 / 1e6,
        "formula": "params × bitwidth × 18 F² / array_efficiency",
    }


def die_area() -> dict:
    nodes = {
        "sky130": {"nm": 130, "sram_um2": 1.896},
        "n28": {"nm": 28, "sram_um2": 0.127},
        "n7": {"nm": 7, "sram_um2": 0.027},
    }
    out = {
        "params": P_27B,
        "model": "Qwen3.8-27B",
        "no_pnr": True,
        "note": (
            "Die is ROM-compiler density × 27B × bit-width. OpenLane never "
            "places the model; it closes one instance of each distinct block."
        ),
        "by_node": {},
        "compiler_18f2": {},
    }
    for name, n in nodes.items():
        out["by_node"][name] = {
            "4bit": rom_die(n["nm"], n["sram_um2"], P_27B, 4),
            "8bit": rom_die(n["nm"], n["sram_um2"], P_27B, 8),
        }
        out["compiler_18f2"][name] = {
            "4bit": rom_die_18f2(n["nm"], P_27B, 4),
            "8bit": rom_die_18f2(n["nm"], P_27B, 8),
        }
    out["by_node"]["n7"]["cap_note"] = (
        "rom_cell_um2() floors FinFET via-ROM at SRAM/4, so 7 nm 4-bit is "
        "1176 mm². compiler_18f2 is the uncapped 18 F² answer (~154 mm²)."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnr", action="store_true")
    parser.add_argument("--ingest", action="store_true", help="reuse openlane/runs/<tag> if present")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[b["id"] for b in BLOCKS],
    )
    args = parser.parse_args()
    wanted = set(args.only) if args.only else {b["id"] for b in BLOCKS}
    by_id = load_existing_blocks()
    for block in BLOCKS:
        if block["id"] not in wanted:
            continue
        by_id[block["id"]] = run_block(block, pnr=args.pnr, ingest=args.ingest)
        mix = mixer_sta()
        if mix:
            rec = by_id.get("mixer") or {
                "id": "mixer",
                "top": "gated_delta_d128_sta",
                "note": BLOCKS[0]["note"],
            }
            rec["sta"] = mix
            by_id["mixer"] = rec
        results = [by_id[b["id"]] for b in BLOCKS if b["id"] in by_id]
        (ROOT / "artifacts" / "sky130_27b_blocks.json").write_text(
            json.dumps(
                {
                    "pdk": "sky130A",
                    "blocks": results,
                    "chip": summarize(results),
                    "die_area_rom": die_area(),
                },
                indent=2,
            )
            + "\n"
        )

    results = [by_id[b["id"]] for b in BLOCKS if b["id"] in by_id]
    die = die_area()
    payload = {
        "pdk": "sky130A",
        "liberty": "sky130_fd_sc_hd tt 25C 1.80V",
        "scale_7nm_f2": F2_SCALE_7NM,
        "scale_7nm_sram_cell": SRAM_SCALE_7NM,
        "fmax_scale_sky130_to_7nm": FMAX_SCALE_7NM,
        "note": (
            "Timing closure is one instance of each distinct 27B block, not the "
            "whole model in OpenLane. SRAM/ROM is a compiler macro; Fmax is the "
            "PE. Die area is ROM density × 27B × bit-width, separately."
        ),
        "blocks": results,
        "chip": summarize(results),
        "die_area_rom": die,
    }
    out = ROOT / "artifacts" / "sky130_27b_blocks.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    chip = payload["chip"]
    print(
        json.dumps(
            {
                "chip_fmax_mhz": chip.get("sky130_chip_fmax_mhz"),
                "fmax_source": chip.get("sky130_fmax_source"),
                "block": chip.get("sky130_chip_fmax_block"),
                "tok_s_4k": chip.get("sky130_tok_s_4k"),
                "estimate_mhz": {
                    "lo": (chip.get("estimate") or {}).get("sky130_mhz_lo"),
                    "nom": (chip.get("estimate") or {}).get("sky130_mhz_nom"),
                    "hi": (chip.get("estimate") or {}).get("sky130_mhz_hi"),
                },
                "fmax_7nm_projected": chip.get("fmax_7nm_mhz_projected"),
                "tok_s_7nm_projected": chip.get("tok_s_7nm_4k_projected"),
                "rom_7nm_4bit_sram_cap_mm2": die["by_node"]["n7"]["4bit"]["mm2"],
                "rom_7nm_4bit_18f2_mm2": die["compiler_18f2"]["n7"]["4bit"]["mm2"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
