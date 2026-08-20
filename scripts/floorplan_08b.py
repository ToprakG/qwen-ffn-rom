#!/usr/bin/env python3
"""0.8B paper floorplan from measured Sky130 tiles (steps 3–7).

Step 8 (magic via-ROM leaf) is skipped — months of layout. Via-ROM mm² stays
the 18 F² estimator. Stdcell mm² is scaled from OpenLane 8×8 CSD / D=4 / D=16.

tok/s is mixer-limited (1-MAC FSM). FFN 8×8 is 1 cycle and does not set the
layer rate. Power at Fmax is OpenSTA@50ns × (Fmax/20 MHz).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLOCK_MEAS_MHZ = 20.0  # OpenSTA power is at 50 ns
F2_SCALE_7NM = (7.0 / 130.0) ** 2
SRAM_UM2 = {"sky130": 1.896, "n7": 0.027}
VIA_ROM_F2 = 18.0
ARRAY_EFF = 0.62


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def um2_per_weight(area_um2: float, n: int) -> float:
    return area_um2 / (n * n)


def p_at_fmax(p_50ns: float | None, fmax_mhz: float | None) -> float | None:
    if p_50ns is None or fmax_mhz is None:
        return None
    return p_50ns * (fmax_mhz / CLOCK_MEAS_MHZ)


def tok_s(fmax_mhz: float | None, cycles: float | None) -> float | None:
    if not fmax_mhz or not cycles:
        return None
    return fmax_mhz * 1e6 / cycles


def nJ(w: float | None, t: float | None) -> float | None:
    if w is None or not t:
        return None
    return (w / t) * 1e9


def via_rom_cell_um2(nm: float, sram_um2: float) -> float:
    f_um = nm / 1000.0
    via = VIA_ROM_F2 * f_um * f_um
    cap = sram_um2 / (8.0 if nm >= 28 else 4.0)
    return max(via, cap)


def fit_quad(d0: float, a0: float, d1: float, a1: float, d: float) -> float:
    """area ≈ a + c D^2 from two measured points."""
    c = (a1 - a0) / (d1 * d1 - d0 * d0)
    a = a0 - c * d0 * d0
    return a + c * d * d


def main() -> None:
    eval_j = load(ROOT / "artifacts" / "eval.json") or {}
    duts = {d["name"]: d for d in eval_j.get("duts") or []}
    d16_eq = load(ROOT / "artifacts" / "delta_d16_equiv.json") or {}
    d16_eda = load(ROOT / "artifacts" / "delta_d16_eda.json") or {}
    dpar_eq = load(ROOT / "artifacts" / "delta_dpar16_equiv.json") or {}
    dpar_eda = load(ROOT / "artifacts" / "delta_dpar16_eda.json") or {}
    d4_eda = load(ROOT / "artifacts" / "delta_d4_eda.json") or {}
    target = load(ROOT / "models" / "target.json") or {}

    tile = duts["ffn_tile_8x8_b4"]
    tap = duts.get("ffn_rom_tap_8x8_b4") or {}
    serial = duts.get("ffn_col_serial_8x8_b4") or {}
    fetch = duts.get("ffn_rom_fetch_8x8_b4") or {}
    d4 = duts["gated_delta_d4"]
    hybrid = duts.get("hybrid_layer_d4_ffn8") or {}

    ol16 = (d16_eda.get("openlane") or {})
    ys16 = (d16_eda.get("yosys") or {})
    ys4 = (d4_eda.get("yosys") or {})
    d16_area = ol16.get("area_um2") or ys16.get("area_um2")
    d16_fmax = ol16.get("fmax_mhz_from_ws") or d4.get("fmax_mhz")
    d16_p = ol16.get("power_w")
    d16_cyc = d16_eq.get("cycles_per_token")
    d16_area_src = "openlane" if ol16.get("area_um2") else "yosys"

    if d16_p is None and d16_area and d4.get("area_um2") and d4.get("power_w"):
        d16_p = d4["power_w"] * (d16_area / d4["area_um2"])

    measured = {
        "clock_power_mhz": CLOCK_MEAS_MHZ,
        "note": "PE numbers from Sky130 OpenLane. nJ uses P(Fmax)=P(50ns)*(Fmax/20).",
        "tiles": {
            "ffn_tile_8x8_b4": {
                "um2": tile["area_um2"],
                "um2_per_weight": um2_per_weight(tile["area_um2"], 8),
                "fmax_mhz": tile["fmax_mhz"],
                "cycles": tile["cycles_per_token"],
                "tok_s": tok_s(tile["fmax_mhz"], tile["cycles_per_token"]),
                "p_50ns_w": tile["power_w"],
                "p_fmax_w": p_at_fmax(tile["power_w"], tile["fmax_mhz"]),
                "power_model": tile.get("power_model"),
            },
            "ffn_rom_tap_8x8_b4": {
                "um2": tap.get("area_um2"),
                "um2_per_weight": um2_per_weight(tap["area_um2"], 8) if tap.get("area_um2") else None,
                "cycles": tap.get("cycles_per_token"),
                "fmax_mhz": tap.get("fmax_mhz"),
            },
            "ffn_col_serial_8x8_b4": {
                "um2": serial.get("area_um2"),
                "um2_per_weight": um2_per_weight(serial["area_um2"], 8) if serial.get("area_um2") else None,
                "cycles": serial.get("cycles_per_token"),
            },
            "ffn_rom_fetch_8x8_b4": {
                "um2": fetch.get("area_um2"),
                "um2_per_weight": um2_per_weight(fetch["area_um2"], 8) if fetch.get("area_um2") else None,
                "cycles": fetch.get("cycles_per_token"),
            },
        },
        "mixers": {
            "gated_delta_d4": {
                "um2": d4["area_um2"],
                "fmax_mhz": d4["fmax_mhz"],
                "cycles": d4["cycles_per_token"],
                "tok_s": tok_s(d4["fmax_mhz"], d4["cycles_per_token"]),
                "p_50ns_w": d4["power_w"],
                "p_fmax_w": p_at_fmax(d4["power_w"], d4["fmax_mhz"]),
                "nJ_fmax": nJ(p_at_fmax(d4["power_w"], d4["fmax_mhz"]), tok_s(d4["fmax_mhz"], d4["cycles_per_token"])),
            },
            "gated_delta_d16": {
                "um2": d16_area,
                "area_source": d16_area_src,
                "fmax_mhz": d16_fmax,
                "fmax_source": "openlane" if ol16.get("fmax_mhz_from_ws") else "d4_fmax_proxy",
                "cycles": d16_cyc,
                "tok_s": tok_s(d16_fmax, d16_cyc),
                "p_50ns_w": d16_p,
                "p_fmax_w": p_at_fmax(d16_p, d16_fmax),
                "nJ_fmax": nJ(p_at_fmax(d16_p, d16_fmax), tok_s(d16_fmax, d16_cyc)),
                "yosys_cells": ys16.get("cell_count"),
            },
            "gated_delta_d16_par": {
                "um2": (dpar_eda.get("yosys") or {}).get("area_um2"),
                "area_source": "yosys",
                "fmax_mhz": d4.get("fmax_mhz"),
                "fmax_source": "d4_fmax_proxy",
                "cycles": dpar_eq.get("cycles_per_token"),
                "tok_s": tok_s(d4.get("fmax_mhz"), dpar_eq.get("cycles_per_token")),
                "mac": dpar_eq.get("mac") or 16,
                "yosys_cells": (dpar_eda.get("yosys") or {}).get("cell_count"),
                "pnr": "skipped_80k_cells",
            },
        },
        "hybrid_layer_d4_ffn8": {
            "um2": hybrid.get("area_um2"),
            "fmax_mhz": hybrid.get("fmax_mhz"),
            "cycles": hybrid.get("cycles_per_token"),
            "tok_s": tok_s(hybrid.get("fmax_mhz"), hybrid.get("cycles_per_token")),
            "p_50ns_w": hybrid.get("power_w"),
            "p_fmax_w": p_at_fmax(hybrid.get("power_w"), hybrid.get("fmax_mhz")),
            "nJ_fmax": nJ(
                p_at_fmax(hybrid.get("power_w"), hybrid.get("fmax_mhz")),
                tok_s(hybrid.get("fmax_mhz"), hybrid.get("cycles_per_token")),
            ),
        },
    }

    # Winning FFN column: word-parallel CSD (step 3).
    u_w = um2_per_weight(tile["area_um2"], 8)
    p_w = tile["power_w"] / 64.0

    h = int(target.get("hidden_size") or 1024)
    inter = int(target.get("intermediate_size") or 3584)
    layers = int(target.get("num_hidden_layers") or 24)
    ffn_w = 3 * h * inter * layers
    sib = target.get("sibling_27b") or {}
    ffn_w_27 = 3 * int(sib.get("hidden_size") or 5120) * int(sib.get("intermediate_size") or 17408) * int(
        sib.get("num_hidden_layers") or 64
    )

    heads = 16
    d_model = 128
    delta_layers = 18
    attn_layers = 6
    kv_heads = 2
    attn_d = 256
    lora_rank = 8
    lora_layers = layers

    # Mixer D=128 from D=4 / D=16 fit (Yosys if no PnR on d16).
    a4 = ys4.get("area_um2") or d4["area_um2"]
    a16 = ys16.get("area_um2") or d16_area
    pe128_ys = fit_quad(4, a4, 16, a16, d_model)
    ol_over_ys = (d4["area_um2"] / a4) if a4 else 1.48
    pe128_um2 = pe128_ys * ol_over_ys
    cyc128 = d16_cyc * (d_model / 16) ** 2
    fmax128 = d16_fmax
    tok128 = tok_s(fmax128, cyc128)
    p128_50 = (d16_p or 0) * (pe128_um2 / (d16_area or 1))
    p128_f = p_at_fmax(p128_50, fmax128)
    # 16 heads in parallel: same cycles, 16× area and power.
    mix_um2 = pe128_um2 * heads
    mix_w = (p128_f or 0) * heads
    mix_tok = tok128  # heads parallel, rate unchanged

    ffn_stdcell_mm2 = ffn_w * u_w / 1e6
    ffn_stdcell_w50 = ffn_w * p_w
    # FFN tiles to hide under mixer cycles: macs/layer / 64 / cycles
    ffn_macs_layer = 3 * h * inter
    tiles_to_hide = max(1, round(ffn_macs_layer / 64.0 / cyc128))
    ffn_live_um2 = tiles_to_hide * tile["area_um2"]
    ffn_live_w = tiles_to_hide * (p_at_fmax(tile["power_w"], tile["fmax_mhz"]) or 0)

    s_bits = delta_layers * heads * d_model * d_model * 16
    s_mm2_sky = s_bits * SRAM_UM2["sky130"] / 1e6
    s_mm2_7 = s_bits * SRAM_UM2["n7"] / 1e6

    def kv_bits(seq: int) -> int:
        return attn_layers * 2 * kv_heads * attn_d * seq * 16

    lora_bits = 2 * h * lora_rank * lora_layers * 16

    via_sky = via_rom_cell_um2(130, SRAM_UM2["sky130"])
    via_7 = via_rom_cell_um2(7, SRAM_UM2["n7"])
    ffn_via_sky = ffn_w * 4 * via_sky / ARRAY_EFF / 1e6
    ffn_via_7 = ffn_w * 4 * via_7 / ARRAY_EFF / 1e6
    ffn_via_27_7 = ffn_w_27 * 4 * via_7 / ARRAY_EFF / 1e6
    ffn_std_7 = ffn_stdcell_mm2 * F2_SCALE_7NM
    mix_7 = mix_um2 / 1e6 * F2_SCALE_7NM

    seqs = (4096, 32768)
    kv = {
        str(s): {
            "bits": kv_bits(s),
            "mm2_sky130": kv_bits(s) * SRAM_UM2["sky130"] / 1e6,
            "mm2_n7": kv_bits(s) * SRAM_UM2["n7"] / 1e6,
        }
        for s in seqs
    }

    sky130_chip = {
        "fits": False,
        "tok_s": mix_tok,
        "watt_fmax": mix_w + ffn_live_w,
        "mm2": {
            "ffn_all_layers_stdcell": ffn_stdcell_mm2,
            "ffn_live_tiles_to_hide_mixer": ffn_live_um2 / 1e6,
            "delta_pes_16x_d128": mix_um2 / 1e6,
            "delta_S_sram": s_mm2_sky,
            "kv_4096": kv["4096"]["mm2_sky130"],
            "kv_32768": kv["32768"]["mm2_sky130"],
            "lora_r8_sram": lora_bits * SRAM_UM2["sky130"] / 1e6,
        },
        "why_not": "FFN as measured CSD is ~134,000 mm²; via-ROM 18 F² is ~500 mm². Neither is a Sky130 die.",
    }
    n7_chip = {
        "fits_reticle": True,
        "tok_s": mix_tok,
        "tok_s_note": "Sky130 Fmax, not a 7 nm STA number. 7 nm would be faster.",
        "watt_fmax": mix_w + ffn_live_w,
        "watt_note": "Sky130 OpenSTA scaled by area; 7 nm watt would drop with Vdd/capacitance.",
        "mm2": {
            "ffn_via_rom_4b": ffn_via_7,
            "ffn_stdcell_f2_scaled": ffn_std_7,
            "delta_pes_f2_scaled": mix_7,
            "delta_S_sram": s_mm2_7,
            "kv_4096": kv["4096"]["mm2_n7"],
            "kv_32768": kv["32768"]["mm2_n7"],
            "lora_r8_sram": lora_bits * SRAM_UM2["n7"] / 1e6,
        },
    }
    n7_chip["mm2"]["total_via_rom_ffn_ctx4k"] = (
        n7_chip["mm2"]["ffn_via_rom_4b"]
        + n7_chip["mm2"]["delta_pes_f2_scaled"]
        + n7_chip["mm2"]["delta_S_sram"]
        + n7_chip["mm2"]["kv_4096"]
        + n7_chip["mm2"]["lora_r8_sram"]
    )
    n7_chip["mm2"]["total_via_rom_ffn_ctx32k"] = (
        n7_chip["mm2"]["ffn_via_rom_4b"]
        + n7_chip["mm2"]["delta_pes_f2_scaled"]
        + n7_chip["mm2"]["delta_S_sram"]
        + n7_chip["mm2"]["kv_32768"]
        + n7_chip["mm2"]["lora_r8_sram"]
    )

    step8 = {
        "status": "skipped",
        "reason": "Magic/GDS via-ROM leaf is months of layout. Step 3 already ranked tap vs fetch as stdcell stand-ins; tap tracks 1-cycle CSD, fetch is the slow ROM.",
    }

    payload = {
        "target": {
            "checkpoint": target.get("checkpoint"),
            "hidden": h,
            "intermediate": inter,
            "layers": layers,
            "delta_layers": delta_layers,
            "attn_layers": attn_layers,
            "linear_heads": heads,
            "linear_d": d_model,
            "attn_q_heads": 8,
            "attn_kv_heads": kv_heads,
            "attn_d": attn_d,
            "ffn_weights": ffn_w,
            "ffn_weights_27b": ffn_w_27,
        },
        "step8_via_rom_gds": step8,
        "measured_um2_per_weight": {
            "wordpar_csd_8x8": u_w,
            "rom_tap_8x8": measured["tiles"]["ffn_rom_tap_8x8_b4"]["um2_per_weight"],
            "serial_csd_8x8": measured["tiles"]["ffn_col_serial_8x8_b4"]["um2_per_weight"],
            "rom_fetch_8x8": measured["tiles"]["ffn_rom_fetch_8x8_b4"]["um2_per_weight"],
            "winner": "wordpar_csd_8x8",
            "note": "stdcell, not via-ROM density",
        },
        "measured": measured,
        "scaled_mixer_d128": {
            "cycles": cyc128,
            "fmax_mhz": fmax128,
            "tok_s_one_head": tok128,
            "um2_one_pe": pe128_um2,
            "p_fmax_w_one_pe": p128_f,
            "heads_parallel": heads,
            "tok_s_layer": mix_tok,
            "um2_16_pes": mix_um2,
            "p_fmax_w_16_pes": mix_w,
            "fit": "Yosys area(D)=a+c D^2 from D=4 and D=16, then OpenLane/Yosys ratio from D=4. Cycles from D=16 *(128/16)^2.",
        },
        "scaled_mixer_d128_dpar": {
            "mac": d_model,
            "cycles": 4 * d_model + 3,
            "fmax_mhz": d4.get("fmax_mhz"),
            "fmax_source": "d4_openlane_proxy",
            "tok_s_one_head": tok_s(d4.get("fmax_mhz"), 4 * d_model + 3),
            "tok_s_layer": tok_s(d4.get("fmax_mhz"), 4 * d_model + 3),
            "heads_parallel": heads,
            "fit": "FSM is 4D+3 (measured 67 at D=16). D=128 with D MACs → 515 cycles. Fmax is D=4 PnR proxy. Area not scaled (only D=16 Yosys).",
            "d16_par_yosys_um2": (dpar_eda.get("yosys") or {}).get("area_um2"),
            "d16_par_yosys_cells": (dpar_eda.get("yosys") or {}).get("cell_count"),
            "pnr": "skipped_d16_par_80k_cells",
        },
        "ffn_live": {
            "tiles_to_hide_under_mixer": tiles_to_hide,
            "um2": ffn_live_um2,
            "p_fmax_w": ffn_live_w,
        },
        "sky130_08b": sky130_chip,
        "n7_08b_paper": n7_chip,
        "n7_27b_estimator": {
            "ffn_via_rom_4b_mm2": ffn_via_27_7,
            "tok_s": mix_tok,
            "tok_s_note": "Same 1-MAC D=128 FSM; more heads still parallel so tok/s does not rise. 27B is estimator-only.",
            "watt_note": "Not quoted. No 27B mixer PnR.",
        },
        "via_rom_cell_um2": {"sky130": via_sky, "n7": via_7},
        "ffn_all_layers_mm2": {
            "sky130_stdcell_csd": ffn_stdcell_mm2,
            "sky130_via_rom_4b": ffn_via_sky,
            "n7_stdcell_f2": ffn_std_7,
            "n7_via_rom_4b": ffn_via_7,
        },
    }

    out = ROOT / "artifacts" / "floorplan_08b.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    m = payload["measured"]
    print("=== measured Sky130 PEs ===")
    print(
        f"  FFN 8x8 CSD   {m['tiles']['ffn_tile_8x8_b4']['tok_s']:.3e} tok/s  "
        f"{m['tiles']['ffn_tile_8x8_b4']['p_fmax_w']*1e3:.2f} mW@Fmax  "
        f"{m['tiles']['ffn_tile_8x8_b4']['um2']/1e6:.4f} mm²"
    )
    print(
        f"  Delta D=4     {m['mixers']['gated_delta_d4']['tok_s']:.3e} tok/s  "
        f"{m['mixers']['gated_delta_d4']['p_fmax_w']*1e3:.2f} mW@Fmax  "
        f"{m['mixers']['gated_delta_d4']['um2']/1e6:.4f} mm²"
    )
    print(
        f"  Delta D=16    {m['mixers']['gated_delta_d16']['tok_s']:.3e} tok/s  "
        f"cyc={d16_cyc}  {d16_area_src} {d16_area/1e6:.4f} mm²"
    )
    dpar_m = m["mixers"].get("gated_delta_d16_par") or {}
    if dpar_m.get("tok_s"):
        print(
            f"  Delta D=16 par {dpar_m['tok_s']:.3e} tok/s  "
            f"cyc={dpar_m.get('cycles')} mac={dpar_m.get('mac')}  "
            f"yosys {(dpar_m.get('um2') or 0)/1e6:.4f} mm²  pnr={dpar_m.get('pnr')}"
        )
    print(
        f"  hybrid layer  {m['hybrid_layer_d4_ffn8']['tok_s']:.3e} tok/s  "
        f"{m['hybrid_layer_d4_ffn8']['p_fmax_w']*1e3:.2f} mW@Fmax  "
        f"{m['hybrid_layer_d4_ffn8']['um2']/1e6:.4f} mm²"
    )
    print("=== 0.8B paper (mixer-limited, 16 heads D=128 1-MAC) ===")
    print(f"  tok/s         {mix_tok:.3e}")
    dpar128 = payload["scaled_mixer_d128_dpar"]
    print("=== 0.8B if mixer is D-parallel (n_mac=D, Fmax proxy) ===")
    print(f"  tok/s         {dpar128['tok_s_layer']:.3e}  cyc={dpar128['cycles']}")
    print(f"  watt @ Fmax   {sky130_chip['watt_fmax']:.2f} W  (Sky130 P scaled; not 7 nm)")
    print(f"  Sky130 FFN CSD mm² {ffn_stdcell_mm2:.0f}  (does not fit)")
    print(f"  7nm via-ROM FFN mm² {ffn_via_7:.2f}")
    print(f"  7nm total ctx4k mm² {n7_chip['mm2']['total_via_rom_ffn_ctx4k']:.2f}")
    print(f"  7nm total ctx32k mm² {n7_chip['mm2']['total_via_rom_ffn_ctx32k']:.2f}")
    print(f"  27B 7nm FFN via-ROM mm² {ffn_via_27_7:.1f}  tok/s {mix_tok:.3e} (same FSM)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
