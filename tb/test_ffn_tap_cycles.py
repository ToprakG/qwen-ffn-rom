"""Count clocks for one 8x8 via-tap handshake (rtl/ffn_tap_unit.v)."""

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
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402


async def reset(dut) -> None:
    dut.en.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_ffn_tap_unit_cycles(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    w = load_w(8).astype(np.int64)
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=8, dtype=np.int64)
    dut.x_flat.value = pack_signed(x, 8)
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
        raise AssertionError("ffn_tap_unit timeout")
    y = unpack_signed(int(dut.y_flat.value), 8, 15)
    np.testing.assert_array_equal(y, w @ x)
    rec = {
        "gate": "ffn_tap_unit y == W_int @ x",
        "status": "PASS",
        "dut": "rtl/ffn_tap_unit.v",
        "cycles_per_tile": cycles,
        "note": "Combo tap is 0 sequential clocks. This is the registered handshake.",
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / "ffn_tap_cycles.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"ffn tap unit cycles={cycles}")
