#!/usr/bin/env python3
"""Honest clocks/token and tok/s vs frequency.

Uses measured:
  artifacts/ffn_tap_cycles.json          8x8 via-tap handshake
  artifacts/attn_decode_cycles.json      D-parallel MAC skeleton (no softmax)
  artifacts/delta_d128_bram_equiv.json   mixer clocks/layer at D=128
  artifacts/decoder_layer_equiv.json     toy full layer (optional)

Does not assume FFN is free. Two FFN schedules:
  serial     one 8x8 tap reused
  hidden     enough taps that FFN finishes inside that layer's mixer/attn window

  python scripts/honest_tok_s.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    p = ROOT / "artifacts" / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run the matching make sim first")
    return json.loads(p.read_text())


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def swiglu_tiles(hidden: int, intermediate: int, tile: int = 8) -> int:
    """gate + up + down, each tiled into tile×tile via-taps."""
    rh, ch = ceil_div(hidden, tile), ceil_div(intermediate, tile)
    # gate, up: (I × H); down: (H × I)
    return 2 * ch * rh + rh * ch


def tok_s(hz: float, cycles: int) -> float:
    return hz / cycles if cycles else 0.0


def main() -> None:
    ffn = load("ffn_tap_cycles.json")
    attn = load("attn_decode_cycles.json")
    mix = load("delta_d128_bram_equiv.json")
    layer = load("decoder_layer_equiv.json") if (ROOT / "artifacts" / "decoder_layer_equiv.json").exists() else {}

    cyc_tile = int(ffn["cycles_per_tile"])
    cyc_mix = int(mix["cycles_per_layer"] or mix["cycles_per_token"])
    slope = float(attn["fit"]["slope_per_S"])
    intercept = float(attn["fit"]["intercept"])

    def attn_layer(seq: int, softmax: bool) -> int:
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

    freqs_hz = [100e6, 200e6, 330e6, 1e9, 1.65e9]
    seqs = (4096, 32768)
    rows = []
    for name, m in models.items():
        tiles = swiglu_tiles(m["hidden"], m["intermediate"])
        cyc_ffn_layer_serial = tiles * cyc_tile
        for seq in seqs:
            for softmax in (False, True):
                cyc_d = m["delta_layers"] * cyc_mix
                cyc_a = m["attn_layers"] * attn_layer(seq, softmax)
                cyc_f_serial = m["layers"] * cyc_ffn_layer_serial
                # Hide FFN under the same-layer mixer or attn window.
                hide_delta = max(1, math.ceil(cyc_ffn_layer_serial / cyc_mix))
                hide_attn = max(1, math.ceil(cyc_ffn_layer_serial / max(attn_layer(seq, softmax), 1)))
                cyc_hidden = cyc_d + cyc_a  # FFN parallel, no extra
                rec = {
                    "model": name,
                    "seq": seq,
                    "softmax_in_attn": softmax,
                    "tiles_per_ffn_layer": tiles,
                    "cycles_ffn_layer_serial": cyc_ffn_layer_serial,
                    "taps_to_hide_under_delta_layer": hide_delta,
                    "taps_to_hide_under_attn_layer": hide_attn,
                    "cycles_delta_all": cyc_d,
                    "cycles_attn_all": cyc_a,
                    "cycles_token_ffn_serial": cyc_d + cyc_a + cyc_f_serial,
                    "cycles_token_ffn_hidden": cyc_hidden,
                    "tok_s_serial": {f"{int(hz/1e6)}MHz": tok_s(hz, cyc_d + cyc_a + cyc_f_serial) for hz in freqs_hz},
                    "tok_s_ffn_hidden": {f"{int(hz/1e6)}MHz": tok_s(hz, cyc_hidden) for hz in freqs_hz},
                }
                rows.append(rec)

    payload = {
        "measured": {
            "ffn_cycles_per_8x8_tile": cyc_tile,
            "mixer_cycles_per_layer_d128": cyc_mix,
            "attn_fit": attn["fit"],
            "attn_measured_small_S": attn["measured"],
            "decoder_layer_toy_cycles": layer.get("cycles_per_layer"),
        },
        "assumptions": [
            "Mixer: measured D=128 BRAM PE, 16/48 heads in parallel (does not cut cycles).",
            "FFN: measured 8x8 via-tap handshake, scaled by tile count of SwiGLU (gate+up+down).",
            "Attn: measured slope vs S on D=4; slope is 2 if one cache row/cycle (D-parallel). Same slope at D=256 if 256 MACs/head and heads in parallel.",
            "Softmax is not in RTL. Rows with softmax_in_attn=true add S cycles per attn layer.",
            "ffn_hidden needs thousands of 8x8 taps on the die (via-ROM story). serial is one reused tap.",
            "Gen-1 10k at 28 nm is only possible in the ffn_hidden column, at ~330 MHz, and still depends on attn at that seq.",
        ],
        "rows": rows,
    }
    out = ROOT / "artifacts" / "honest_tok_s.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("measured  8x8 tap", cyc_tile, "clk   mixer D=128", cyc_mix, "clk/layer")
    print("attn      cycles =", f"{intercept:.2f} + {slope:.3f} * S   (no softmax)")
    print()
    print(f"{'model':8} {'seq':6} {'sm':3} {'cyc serial':>12} {'cyc hidden':>12} {'10k@330M hidden':>16} {'50k@1.65G hidden':>18}")
    for r in rows:
        hz330 = r["tok_s_ffn_hidden"]["330MHz"]
        hz165 = r["tok_s_ffn_hidden"]["1650MHz"]
        print(
            f"{r['model']:8} {r['seq']:6} {int(r['softmax_in_attn']):3} "
            f"{r['cycles_token_ffn_serial']:12} {r['cycles_token_ffn_hidden']:12} "
            f"{hz330:16.1f} {hz165:18.1f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
