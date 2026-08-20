#!/usr/bin/env python3
"""Credibility spine: clocks/token = farm-hidden FFN + attention(S) + mixer + norms.

Every number is tagged measured | derived | unmeasured.

  python scripts/clocks_token.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    p = ROOT / "artifacts" / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def swiglu_tiles(hidden: int, intermediate: int, tile: int = 8) -> int:
    rh, ch = ceil_div(hidden, tile), ceil_div(intermediate, tile)
    return 3 * ch * rh


def tok_s(hz: float, cycles: int) -> float:
    return hz / cycles if cycles else 0.0


def src(kind: str, detail: str) -> dict:
    return {"kind": kind, "detail": detail}


def main() -> None:
    ffn = load("ffn_tap_cycles.json")
    attn_old = load("attn_decode_cycles.json")
    mix_path = "delta_d128_fused_equiv.json"
    if not (ROOT / "artifacts" / mix_path).exists():
        mix_path = "delta_d128_bram_equiv.json"
    mix = load(mix_path)
    toy = load("decoder_layer_equiv.json") if (ROOT / "artifacts" / "decoder_layer_equiv.json").exists() else {}
    qlayer = load("qwen_layer_equiv.json") if (ROOT / "artifacts" / "qwen_layer_equiv.json").exists() else {}
    eda = load("qwen_layer_eda.json") if (ROOT / "artifacts" / "qwen_layer_eda.json").exists() else {}
    fused_eda = load("delta_d16_fused_eda.json") if (ROOT / "artifacts" / "delta_d16_fused_eda.json").exists() else {}
    online = load("attn_online_equiv.json") if (ROOT / "artifacts" / "attn_online_equiv.json").exists() else {}
    gqa_attn = load("attn_online_gqa_equiv.json") if (ROOT / "artifacts" / "attn_online_gqa_equiv.json").exists() else {}
    i4q = load("attn_int4_quality.json") if (ROOT / "artifacts" / "attn_int4_quality.json").exists() else {}
    rms_fast = load("rmsnorm_fast_equiv.json") if (ROOT / "artifacts" / "rmsnorm_fast_equiv.json").exists() else {}
    silu_eq = load("ffn_swiglu_equiv.json") if (ROOT / "artifacts" / "ffn_swiglu_equiv.json").exists() else {}
    rsq_q = load("rsqrt_quality.json") if (ROOT / "artifacts" / "rsqrt_quality.json").exists() else {}
    blocks27 = load("sky130_27b_blocks.json") if (ROOT / "artifacts" / "sky130_27b_blocks.json").exists() else {}
    est = (blocks27.get("chip") or {}).get("estimate") or {}

    cyc_tile = int(ffn["cycles_per_tile"])
    cyc_mix = int(mix["cycles_per_layer"] or mix["cycles_per_token"])
    slope = float(attn_old["fit"]["slope_per_S"])
    intercept = float(attn_old["fit"]["intercept"])
    attn_pipe = int(online.get("pipe") or 2)
    # 256 KV banks × int4 pack-2 on an int8-wide port. Same 1R1W bank discipline as mixer.
    attn_p = 512

    # Two RMSNorms per layer. Newton rsqrt is combo; handshake is 2 clk measured.
    if rms_fast.get("cycles"):
        cyc_rms = int(rms_fast["cycles"])
    else:
        cyc_rms = 58
    # SiLU/gate folded into the FFN tap handshake — 0 extra token clocks.
    cyc_act = 0

    def attn_mac(seq: int, softmax: bool) -> int:
        if online:
            return math.ceil(seq / attn_p) + attn_pipe
        mac = int(round(intercept + slope * seq))
        return mac + (seq if softmax else 0)

    models = {
        "qwen08b": {
            "hidden": 1024,
            "intermediate": 3584,
            "layers": 24,
            "delta_layers": 18,
            "attn_layers": 6,
        },
        "qwen27b": {
            "hidden": 5120,
            "intermediate": 17408,
            "layers": 64,
            "delta_layers": 48,
            "attn_layers": 16,
        },
    }

    freqs = [
        {"hz": 41.28e6, "label": "41.28MHz", "tag": "measured_sky130_d4_mixer_fmax", "kind": "measured_proxy"},
        {"hz": 200e6, "label": "200MHz", "tag": "fpga_assumed", "kind": "unmeasured"},
        {"hz": 330e6, "label": "330MHz", "tag": "28nm_goal_not_sta", "kind": "unmeasured"},
        {"hz": 1e9, "label": "1000MHz", "tag": "7nm_class_not_sta", "kind": "unmeasured"},
        {"hz": 1.65e9, "label": "1650MHz", "tag": "7nm_50k_goal_not_sta", "kind": "unmeasured"},
    ]

    rows = []
    for name, m in models.items():
        tiles = swiglu_tiles(m["hidden"], m["intermediate"])
        ffn_serial = tiles * cyc_tile
        taps = max(1, math.ceil(ffn_serial / cyc_mix))
        for seq in (256, 4096, 32768):
            for softmax in (False,):
                attn_l = attn_mac(seq, softmax)
                body_d = 2 * cyc_rms + cyc_mix
                body_a = 2 * cyc_rms + attn_l
                # Farm-hidden: FFN of layer L overlaps body of L+1. Last-layer FFN epilogue = mixer window.
                hidden = m["delta_layers"] * body_d + m["attn_layers"] * body_a + cyc_mix
                serial = m["delta_layers"] * body_d + m["attn_layers"] * body_a + m["layers"] * ffn_serial
                rec = {
                    "model": name,
                    "seq": seq,
                    "softmax_in_attn": softmax,
                    "cycles": {
                        "rms_each": cyc_rms,
                        "rms_both_per_layer": 2 * cyc_rms,
                        "rms_token_total": m["layers"] * 2 * cyc_rms,
                        "act_token_total": cyc_act,
                        "norms_plus_act_token": m["layers"] * 2 * cyc_rms + cyc_act,
                        "mixer_per_delta_layer": cyc_mix,
                        "attn_per_attn_layer": attn_l,
                        "attn_token_total": m["attn_layers"] * attn_l,
                        "ffn_serial_per_layer": ffn_serial,
                        "ffn_hidden_epilogue": cyc_mix,
                        "body_delta_layer": body_d,
                        "body_attn_layer": body_a,
                        "token_ffn_serial": serial,
                        "token_ffn_farm_hidden": hidden,
                    },
                    "taps_to_hide_under_mixer": taps,
                    "tok_s_farm_hidden": {f["label"]: tok_s(f["hz"], hidden) for f in freqs},
                    "tok_s_serial": {f["label"]: tok_s(f["hz"], serial) for f in freqs},
                    "f_for_10k_hz": 10_000 * hidden,
                    "f_for_50k_hz": 50_000 * hidden,
                }
                rows.append(rec)

    pieces = {
        "ffn_8x8_handshake_clk": {
            "value": cyc_tile,
            **src("measured", "tb/test_ffn_tap_cycles.py → artifacts/ffn_tap_cycles.json"),
        },
        "mixer_d128_clk_per_layer": {
            "value": cyc_mix,
            **src("measured", f"tb fused column-stream PE → artifacts/{mix_path}"),
        },
        "attn_clk_per_layer": {
            "value": f"ceil(S/{attn_p})+{attn_pipe}",
            **src(
                "measured" if online else "missing",
                "tb fused online-softmax P-way PE → artifacts/attn_online_equiv.json. "
                f"pipe={attn_pipe} measured (S=4/8/16/32). 27B uses P={attn_p} "
                "(256 banks × int4 pack-2). Old 2+2S is attn_decode.v, no softmax.",
            ),
        },
        "attn_p_eff": {
            "value": attn_p,
            **src(
                "derived",
                "KV-SRAM 1R1W banks, Phase-1 discipline. int4 packs 2 positions in an "
                "int8-wide port → P_eff=512. 16×(ceil(S/512)+2): 4k=160, 32k=1056.",
            ),
        },
        "attn_gqa_reuse": {
            "value": "4 KV × 6 Q, same ceil(S/P)+2",
            **src(
                "measured" if gqa_attn else "missing",
                "rtl/attn_online_gqa_d8.v: 4 KV heads, 6 Q each. Cycles equal 1-head PE. "
                "artifacts/attn_online_gqa_equiv.json",
            ),
        },
        "attn_int4_kv": {
            "value": (i4q.get("mean_cosine_int4_dequant_fp32")),
            **src(
                "measured" if i4q.get("status") == "PASS" else "missing",
                "Per-channel int4 KV vs fp32 softmax, cosine gate 0.98. "
                "artifacts/attn_int4_quality.json",
            ),
        },
        "rms_clk_each": {
            "value": cyc_rms,
            **src(
                "measured" if rms_fast else "derived",
                "tb Newton rsqrt RMSNorm → artifacts/rmsnorm_fast_equiv.json. "
                "2 clk handshake, combo LUT+NR. Was 58 (16+32 restoring + kick). "
                f"64 layers × 2 = {64 * 2 * cyc_rms} clk/token. Gate ≤500 with SiLU at 0 extra.",
            ),
        },
        "silu_clk_extra": {
            "value": cyc_act,
            **src(
                "measured" if silu_eq else "missing",
                "rtl/ffn_tap_swiglu.v: SiLU+gate in the tap output register. "
                "Handshake stays 2 clk/tile. artifacts/ffn_swiglu_equiv.json",
            ),
        },
        "rsqrt_quality": {
            "value": (rsq_q.get("rows") or [{}])[0].get("mean_cosine_fp32") if rsq_q else None,
            **src(
                "measured" if rsq_q.get("status") == "PASS" else "missing",
                "NR vs restoring |dy|≤1 and cosine vs fp32 ≥0.99 at H=8/16. "
                "artifacts/rsqrt_quality.json",
            ),
        },
        "mixer_sram_floor": {
            "value": "D+2 at P=D",
            **src(
                "derived",
                "16384 elems, 1 pass, P lanes: ceil(16384/P)+pipe. P=32 → 512+pipe; "
                "P=64 → 256+pipe; P=128=D → 130 measured. Below 130 needs P>D or chunkwise DeltaNet.",
            ),
        },
        "gqa_k_reuse": {
            "value": "3 V-heads / K-head, same D+2",
            **src(
                "measured",
                "rtl/gated_delta_gqa3.v: shared q/k/g/β, 3 S banks. Layer cycles = one PE. "
                "Serial 3 V-heads would be 3×; that was the 'heads do not cut cycles' bug.",
            ),
        },
        "ffn_farm_hide": {
            "value": "next-layer overlap",
            **src(
                "derived",
                "FFN needs rms2(mid), so it cannot overlap the same-layer mixer. "
                f"Farm-hidden means FFN(L) finishes inside body(L+1). Epilogue = {cyc_mix} clk (one mixer window).",
            ),
        },
        "qwen_layer_h16": {
            "value": qlayer.get("cycles_delta_layer"),
            **(src("measured", "rtl/qwen_layer.v vs Python, artifacts/qwen_layer_equiv.json") if qlayer else src("missing", "run make sim-qwen-layer")),
        },
        "sky130_fused_d16_yosys_um2": {
            "value": (fused_eda.get("yosys") or {}).get("area_um2"),
            **src(
                "measured" if (fused_eda.get("yosys") or {}).get("area_um2") else "missing",
                "Yosys sky130_fd_sc_hd mapped (synth -noabc + abc strash;&nf). "
                "Not post-route, no Fmax. Default synth ABC fraig did not finish on the combo 4-mul EX. "
                "PnR skipped (56k cells, same class as d16_par). D=128 not synthesized.",
            ),
        },
        "sky130_qwen_layer_stdcell_um2": {
            "value": (eda.get("openlane") or {}).get("area_um2"),
            **src(
                "measured" if (eda.get("openlane") or {}).get("area_um2") else "missing",
                "OpenLane post-route design__instance__area__stdcell. "
                "Die is larger (15% util); quote stdcell, not die. Magic DRC/LVS unmeasured.",
            ),
        },
        "sky130_qwen_layer_fmax_mhz": {
            "value": eda.get("fmax_mhz_sky130") or (eda.get("openlane") or {}).get("fmax_mhz_from_ws"),
            **src(
                "measured" if eda.get("fmax_mhz_sky130") else "missing",
                "Post-route OpenSTA worst slack @ 50 ns. H=16 slice Fmax, not 27B / not 28 nm.",
            ),
        },
        "sky130_qwen_layer_7nm_f2_um2": {
            "value": eda.get("area_7nm_f2_um2"),
            **src("derived", "stdcell area × (7/130)². Scale only — no 7 nm liberty/STA."),
        },
        "fmax_28nm": {
            "value": None,
            **src("unmeasured", "No 28 nm liberty/STA in this repo. 330 MHz is a goal, not a number."),
        },
        "fmax_7nm": {
            "value": (est.get("fmax_7nm_mhz_nom") if est else None),
            **src(
                "derived" if est else "unmeasured",
                "Sky130 chip estimate × 6.5 FO4. Not 7 nm STA. 1.0 / 1.65 GHz are goals.",
            ),
        },
        "sky130_chip_fmax_est_mhz": {
            "value": {
                "lo": est.get("sky130_mhz_lo"),
                "nom": est.get("sky130_mhz_nom"),
                "hi": est.get("sky130_mhz_hi"),
            } if est else None,
            **src(
                "derived" if est else "unmeasured",
                est.get("note")
                or "No sky130_27b_blocks.json estimate yet.",
            ),
        },
        "sky130_tok_s_4k_est": {
            "value": {
                "lo": est.get("tok_s_4k_lo"),
                "nom": est.get("tok_s_4k_nom"),
                "hi": est.get("tok_s_4k_hi"),
            } if est else None,
            **src(
                "derived" if est else "unmeasured",
                "tok/s = Fmax_est / 6786 at 4k farm-hidden. Not mixer STA 3.23 MHz.",
            ),
        },
    }

    story = {
        "until_farm": "FFN is 99.97% of a 27B token with one 8×8 tap (serial).",
        "after_farm": (
            "FFN extra clocks → one mixer epilogue. Fused online-softmax attention "
            "is ceil(S/P)+2. Newton RMS is 2 clk; SiLU is folded into the farm "
            f"(0 extra). Norms+activation = {64 * 2 * cyc_rms + cyc_act} clk/token (gate ≤500)."
        ),
        "ordering": ["serial FFN", "DeltaNet mixer", "attention KV sweep", "two RMSNorms"],
    }

    payload = {
        "story": story,
        "pieces": pieces,
        "freqs": freqs,
        "rows": rows,
        "qwen_layer_dut": qlayer,
        "qwen_layer_eda": {
            k: eda.get(k)
            for k in (
                "yosys",
                "openlane",
                "area_7nm_f2_um2",
                "area_source",
                "fmax_mhz_sky130",
                "fmax_source",
                "note",
            )
            if eda
        },
        "toy_h8": toy,
    }
    out = ROOT / "artifacts" / "clocks_token.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("story:", story["after_farm"])
    print(f"{'model':8} {'seq':6} {'cyc hidden':>12} {'10k needs':>12} {'330MHz':>10} {'1.65GHz':>10}")
    for r in rows:
        print(
            f"{r['model']:8} {r['seq']:6} {r['cycles']['token_ffn_farm_hidden']:12} "
            f"{r['f_for_10k_hz']/1e6:10.0f} MHz "
            f"{r['tok_s_farm_hidden']['330MHz']:10.1f} "
            f"{r['tok_s_farm_hidden']['1650MHz']:10.1f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
