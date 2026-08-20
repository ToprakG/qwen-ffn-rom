"""Xilinx FPGA synth for the BRAM Gated DeltaNet PE.

Yosys `synth_xilinx -family xc7` on this machine. No Fmax until Vivado or
nextpnr-xilinx; tok/s uses an assumed 200 MHz Artix-7 clock, labeled.

  python scripts/eda_fpga.py            # D=4,16,128 + D=4 farm
  python scripts/eda_fpga.py --d 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPGA_FMAX_MHZ = 200.0  # assumed 7-series until Vivado
MODEL_LAYERS_08B = 24
MODEL_LAYERS_27B = 64

DUTS = {
    4: {
        "top": "gated_delta_d4_bram",
        "rtl": [
            ROOT / "rtl" / "delta_s_col_ram.v",
            ROOT / "rtl" / "gated_delta_bram.v",
            ROOT / "rtl" / "gated_delta_d4_bram.v",
        ],
        "mac": 4,
        "n_layers": 1,
    },
    16: {
        "top": "gated_delta_d16_bram",
        "rtl": [
            ROOT / "rtl" / "delta_s_col_ram.v",
            ROOT / "rtl" / "gated_delta_bram.v",
            ROOT / "rtl" / "gated_delta_d16_bram.v",
        ],
        "mac": 16,
        "n_layers": 1,
    },
    128: {
        "top": "gated_delta_d128_bram",
        "rtl": [
            ROOT / "rtl" / "delta_s_col_ram.v",
            ROOT / "rtl" / "gated_delta_bram.v",
            ROOT / "rtl" / "gated_delta_d128_bram.v",
        ],
        "mac": 128,
        "n_layers": 1,
    },
}


def parse_xilinx_stat(text: str) -> dict:
    # Cell library dump also says "LUT2"; only parse the final `stat` block.
    idx = text.rfind("Printing statistics.")
    chunk = text[idx:] if idx >= 0 else text[-4000:]

    def grab(pat: str) -> int | None:
        m = re.search(pat, chunk, re.M)
        return int(m.group(1).replace(",", "")) if m else None

    lut = 0
    for k in range(1, 7):
        lut += grab(rf"^\s+(\d+)\s+LUT{k}\b") or 0
    return {
        "lut": lut or None,
        "ff": grab(r"^\s+(\d+)\s+FDRE\b"),
        "dsp": grab(r"^\s+(\d+)\s+DSP48E1\b") or grab(r"^\s+(\d+)\s+DSP48E2\b"),
        "ramb18": grab(r"^\s+(\d+)\s+RAMB18E1\b") or 0,
        "ramb36": grab(r"^\s+(\d+)\s+RAMB36E1\b") or 0,
        "bram18_equiv": (grab(r"^\s+(\d+)\s+RAMB18E1\b") or 0)
        + 2 * (grab(r"^\s+(\d+)\s+RAMB36E1\b") or 0),
        "cell_count": grab(r"^\s+(\d+)\s+cells\b"),
        "carry4": grab(r"^\s+(\d+)\s+CARRY4\b"),
    }


def yosys_xilinx(rtl: list[Path], top: str, rpt: Path) -> dict:
    reads = "\n".join(f"read_verilog -sv {p.as_posix()}" for p in rtl)
    ys = f"""
{reads}
hierarchy -check -top {top}
proc; flatten
opt_expr; opt_clean
synth_xilinx -family xc7 -top {top}
stat
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
        raise SystemExit(f"yosys synth_xilinx failed for {top}:\n{proc.stderr[-3000:]}")
    stats = parse_xilinx_stat(combined)
    stats["rtl"] = [str(p.relative_to(ROOT)) for p in rtl]
    stats["top"] = top
    stats["family"] = "xc7"
    stats["tool"] = "yosys synth_xilinx"
    return stats


def model_tok_s(cycles_per_layer: int | None, n_layers: int, fmax_mhz: float) -> float | None:
    if not cycles_per_layer:
        return None
    return (fmax_mhz * 1e6) / (n_layers * cycles_per_layer)


def attach_equiv(payload: dict, d: int, n_layers: int) -> None:
    if n_layers > 1:
        eq = json.loads((ROOT / "artifacts" / "qwen08b_farm_d4_equiv.json").read_text()) if (
            ROOT / "artifacts" / "qwen08b_farm_d4_equiv.json"
        ).exists() else {}
    else:
        path = ROOT / "artifacts" / f"delta_d{d}_bram_equiv.json"
        eq = json.loads(path.read_text()) if path.exists() else {}
    cyc = eq.get("cycles_per_layer") or eq.get("cycles_per_token")
    payload["equiv_status"] = eq.get("status")
    payload["cycles_per_layer"] = cyc
    payload["cycles_per_token"] = eq.get("cycles_per_token")
    payload["cycles_per_model_token_08b"] = eq.get("cycles_per_model_token") or (
        cyc * MODEL_LAYERS_08B if cyc else None
    )
    payload["cycles_per_model_token_27b"] = cyc * MODEL_LAYERS_27B if cyc else None
    payload["fmax_mhz"] = FPGA_FMAX_MHZ
    payload["fmax_source"] = "assumed_200mhz_xc7"
    payload["model_tok_s_08b"] = model_tok_s(cyc, MODEL_LAYERS_08B, FPGA_FMAX_MHZ)
    payload["model_tok_s_27b"] = model_tok_s(cyc, MODEL_LAYERS_27B, FPGA_FMAX_MHZ)
    payload["layer_tok_s"] = (FPGA_FMAX_MHZ * 1e6 / cyc) if cyc else None


def synth_one(d: int, farm: bool = False) -> dict:
    if farm:
        top = "qwen08b_farm_d4"
        rtl = [
            ROOT / "rtl" / "delta_s_col_ram.v",
            ROOT / "rtl" / "gated_delta_bram.v",
            ROOT / "rtl" / "qwen08b_delta_farm.v",
            ROOT / "rtl" / "qwen08b_farm_d4.v",
        ]
        tag = "qwen08b_farm_d4"
        n_layers = 24
        mac = 4
        d = 4
    else:
        spec = DUTS[d]
        top = spec["top"]
        rtl = spec["rtl"]
        tag = top
        n_layers = spec["n_layers"]
        mac = spec["mac"]
    rpt = ROOT / "artifacts" / "eda" / f"yosys_xilinx_{tag}.rpt"
    print(f"=== yosys synth_xilinx {top} D={d} ===", flush=True)
    ys = yosys_xilinx(rtl, top, rpt)
    payload = {
        "platform": "fpga_xc7",
        "dut": top,
        "d": d,
        "mac": mac,
        "n_layers": n_layers,
        "note": (
            "BRAM S, one DSP48 per lane (25×18 muxed 16×8 / 8×25), overlapped BRAM "
            "rows (~4D+9 cyc/layer, not 12D). 0.8B walks 24 layers in time on one farm. "
            "Fmax is assumed 200 MHz until Vivado. Headline is model tok/s."
        ),
        "yosys_xilinx": ys,
    }
    attach_equiv(payload, d, n_layers)
    out = ROOT / "artifacts" / f"{tag}_fpga.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, choices=(4, 16, 128))
    parser.add_argument("--farm", action="store_true")
    args = parser.parse_args()
    if args.farm:
        synth_one(4, farm=True)
        return
    ds = (args.d,) if args.d else (4, 16, 128)
    for d in ds:
        synth_one(d)
    if args.d is None:
        synth_one(4, farm=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
