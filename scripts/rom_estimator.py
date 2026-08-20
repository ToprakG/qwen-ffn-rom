#!/usr/bin/env python3
"""Estimate via-ROM capacity for Qwen3.5 FFN weights on one reticle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ASML scanner max exposure field (mm).
RETICLE_MM = (26.0, 33.0)
ARRAY_EFFICIENCY = 0.62

# 0.8B vs 27B measured FFN share of total params (tied embeddings).
FFN_FRAC_08B = 0.330
FFN_FRAC_27B = 0.634
P_08B = 0.8e9
P_27B = 27e9

BITWIDTHS = (1, 2, 4, 8, 16)

NODES = (
    {"id": "sky130", "name": "Sky130 (open)", "nm": 130, "sram_um2": 1.896},
    {"id": "ihp130", "name": "IHP SG13G2", "nm": 130, "sram_um2": 1.50},
    {"id": "n65", "name": "65 nm", "nm": 65, "sram_um2": 0.50},
    {"id": "n28", "name": "28 nm", "nm": 28, "sram_um2": 0.127},
    {"id": "n16", "name": "16 nm FinFET", "nm": 16, "sram_um2": 0.070},
    {"id": "n7", "name": "7 nm FinFET", "nm": 7, "sram_um2": 0.027},
    {"id": "n5", "name": "5 nm FinFET", "nm": 5, "sram_um2": 0.021},
)


def load_target(path: Path) -> dict:
    return json.loads(path.read_text())


def ffn_params_per_layer(hidden: int, intermediate: int) -> int:
    return 3 * hidden * intermediate


def rom_cell_um2(nm: float, sram_um2: float) -> float:
    """Via-ROM ~18 F², capped so FinFET density does not beat SRAM by >4–8×."""
    f_um = nm / 1000.0
    via_rom = 18.0 * f_um * f_um
    sram_ratio = 8.0 if nm >= 28 else 4.0
    return max(via_rom, sram_um2 / sram_ratio)


def reticle_um2() -> float:
    w_mm, h_mm = RETICLE_MM
    return w_mm * h_mm * 1e6


def bits_on_reticle(cell_um2: float) -> float:
    return reticle_um2() * ARRAY_EFFICIENCY / cell_um2


def ffn_fraction(total_params: float) -> float:
    span = math.log(P_27B) - math.log(P_08B)
    t = (math.log(max(total_params, 1.0)) - math.log(P_08B)) / span
    t = min(1.0, max(0.0, t))
    return FFN_FRAC_08B + t * (FFN_FRAC_27B - FFN_FRAC_08B)


def equiv_dense_params(ffn_capacity: float) -> float:
    p = ffn_capacity / 0.5
    for _ in range(10):
        p = ffn_capacity / ffn_fraction(p)
    return p


def fmt_params(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:.0f}"


def estimate(target: dict) -> dict:
    h = int(target["hidden_size"])
    i = int(target["intermediate_size"])
    layers = int(target["num_hidden_layers"])
    matrix = int(target["target_weights"])
    ffn_layer = ffn_params_per_layer(h, i)
    ffn_model = ffn_layer * layers

    nodes = []
    for node in NODES:
        cell = rom_cell_um2(node["nm"], node["sram_um2"])
        bits = bits_on_reticle(cell)
        by_bw = []
        for bw in BITWIDTHS:
            cap = bits / bw
            by_bw.append(
                {
                    "bitwidth": bw,
                    "ffn_weights_fit": cap,
                    "equiv_dense_params": equiv_dense_params(cap),
                    "target_matrix_fit": cap >= matrix,
                    "full_08b_ffn_fit": cap >= ffn_model,
                    "target_matrix_copies": cap / matrix,
                    "full_08b_ffn_copies": cap / ffn_model,
                    "target_array_mm2": matrix * bw * cell / 1e6 / ARRAY_EFFICIENCY,
                    "target_array_mm2_stdcell": matrix * 509.375 / 1e6,
                    "full_08b_ffn_mm2_stdcell": ffn_model * 509.375 / 1e6,
                }
            )
        nodes.append(
            {
                **node,
                "rom_cell_um2": cell,
                "rom_vs_sram": node["sram_um2"] / cell,
                "bits": bits,
                "gbit": bits / 1e9,
                "by_bitwidth": by_bw,
            }
        )

    return {
        "assumptions": {
            "reticle_mm": list(RETICLE_MM),
            "reticle_mm2": RETICLE_MM[0] * RETICLE_MM[1],
            "array_efficiency": ARRAY_EFFICIENCY,
            "rom_model": "via-programmed ROM at 18 F², but not denser than SRAM/8 (≥28 nm) or SRAM/4 (FinFET)",
            "stdcell_um2_per_weight": 509.375,
            "stdcell_source": "OpenLane sky130_fd_sc_hd ffn_tile_8x8_b4 32600 µm² / 64 weights. Not via-ROM.",
        },
        "target": {
            "checkpoint": target["checkpoint"],
            "tensor": target["target_tensor"],
            "shape": target["target_shape"],
            "weights": matrix,
            "ffn_per_layer": ffn_layer,
            "ffn_all_layers": ffn_model,
            "ffn_fraction_08b": ffn_model / target["params_total"],
        },
        "nodes": nodes,
    }


def print_table(report: dict) -> None:
    print(
        f"Reticle {report['assumptions']['reticle_mm'][0]:.0f}×"
        f"{report['assumptions']['reticle_mm'][1]:.0f} mm, "
        f"array efficiency {report['assumptions']['array_efficiency']:.0%}"
    )
    t = report["target"]
    print(
        f"Target {t['tensor']}  {t['shape'][0]}×{t['shape'][1]}  "
        f"({t['weights']/1e6:.2f}M weights)"
    )
    header = f"{'node':<18} {'cell um2':>10} {'Gbit':>8}"
    for bw in BITWIDTHS:
        header += f" {('P@' + str(bw) + 'b'):>10}"
    print(header)
    for node in report["nodes"]:
        row = f"{node['name']:<18} {node['rom_cell_um2']:10.4f} {node['gbit']:8.2f}"
        for cell in node["by_bitwidth"]:
            row += f" {fmt_params(cell['equiv_dense_params']):>10}"
        print(row)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=root / "models" / "target.json",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=root / "models" / "rom_estimate.json",
    )
    args = parser.parse_args()
    report = estimate(load_target(args.target))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print_table(report)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
