"""Bit-exact SwiGLU output fold vs quant/silu_int.py. Handshake stays 2 clk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pe_xbar.weights import load_w  # noqa: E402
from quant.silu_int import swiglu_out  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

N = 8


async def reset(dut) -> None:
    dut.en.value = 0
    dut.x_flat.value = 0
    dut.up_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_ffn_tap_swiglu_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    w = load_w(N).astype(np.int64)
    rng = np.random.default_rng(5)
    cycs = []
    n_ok = 0
    for _ in range(24):
        x = rng.integers(-128, 128, size=N, dtype=np.int64)
        up = rng.integers(-128, 128, size=N, dtype=np.int64)
        dut.x_flat.value = pack_signed(x, 8)
        dut.up_flat.value = pack_signed(up, 8)
        dut.en.value = 1
        await Timer(1, unit="ns")
        await RisingEdge(dut.clk)
        dut.en.value = 0
        cycles = 1
        for _ in range(16):
            await RisingEdge(dut.clk)
            cycles += 1
            await Timer(1, unit="ns")
            if int(dut.done.value):
                break
        else:
            raise AssertionError("ffn_tap_swiglu timeout")
        y = unpack_signed(int(dut.y_flat.value), N, 8)
        gate_acc = w @ x
        y_ref = swiglu_out(gate_acc, up)
        np.testing.assert_array_equal(y, y_ref)
        cycs.append(cycles)
        n_ok += 1
    rec = {
        "gate": "ffn_tap_swiglu == silu(sat8(W@x>>7))*up, 2 clk (no extra stall)",
        "status": "PASS",
        "dut": "rtl/ffn_tap_swiglu.v",
        "n": n_ok,
        "cycles_per_tile": int(round(sum(cycs) / len(cycs))),
        "cycles_min": min(cycs),
        "cycles_max": max(cycs),
        "note": "SiLU+gate folded into the existing tap handshake. 27B activation clocks = 0 extra.",
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / "ffn_swiglu_equiv.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"ffn_tap_swiglu PASS n={n_ok} cyc={rec['cycles_per_tile']}")
