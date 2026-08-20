"""Bit-exact fast RMSNorm (Newton rsqrt) vs quant/rsqrt_int.py."""

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

from quant.rsqrt_int import rmsnorm_nr  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

H = 8


async def reset(dut) -> None:
    dut.en.value = 0
    dut.x_flat.value = 0
    dut.w_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_once(dut, x, w) -> tuple[np.ndarray, int]:
    dut.x_flat.value = pack_signed(x, 8)
    dut.w_flat.value = pack_signed(w, 8)
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
            y = unpack_signed(int(dut.y_flat.value), H, 8)
            return y, cycles
    raise AssertionError("rmsnorm timeout")


@cocotb.test()
async def test_rmsnorm_fast_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(4)
    measured = []
    n_ok = 0
    cases = [
        np.zeros(H, dtype=np.int64),
        np.ones(H, dtype=np.int64),
        np.full(H, 127, dtype=np.int64),
        np.full(H, -128, dtype=np.int64),
    ]
    for x in cases:
        w = np.full(H, 64, dtype=np.int64)
        y, cyc = await run_once(dut, x, w)
        np.testing.assert_array_equal(y, rmsnorm_nr(x, w))
        measured.append({"kind": "edge", "cycles": cyc})
        n_ok += 1
    for _ in range(48):
        x = rng.integers(-128, 128, size=H, dtype=np.int64)
        w = rng.integers(-128, 128, size=H, dtype=np.int64)
        y, cyc = await run_once(dut, x, w)
        np.testing.assert_array_equal(y, rmsnorm_nr(x, w))
        measured.append({"kind": "rand", "cycles": cyc})
        n_ok += 1
    cycs = [m["cycles"] for m in measured]
    rec = {
        "gate": "fused Newton rsqrt RMSNorm == rmsnorm_nr, 2 clk",
        "status": "PASS",
        "dut": "rtl/rmsnorm8.v",
        "h": H,
        "n": n_ok,
        "cycles": int(round(sum(cycs) / len(cycs))),
        "cycles_min": min(cycs),
        "cycles_max": max(cycs),
        "formula": "cycles = 2 (capture + mul/done); inv is combo NR",
        "note": "Replaces 16+32 restoring isqrt/div. Width-independent ssq+NR.",
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / "rmsnorm_fast_equiv.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"rmsnorm_fast PASS n={n_ok} cyc={rec['cycles']}")
