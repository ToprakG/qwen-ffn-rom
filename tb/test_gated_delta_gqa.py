"""Bit-exact 1 K-head × 3 V-heads vs three independent Python PEs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.delta_int import O_W, QK_W, SHIFT, V_W, gated_delta_step  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

D = int(os.environ.get("DELTA_D", "16"))
N_V = 3


async def reset(dut) -> None:
    dut.en.value = 0
    dut.q_flat.value = 0
    dut.k_flat.value = 0
    dut.v_flat.value = 0
    dut.g.value = 0
    dut.beta.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    for _ in range(D + 64):
        await RisingEdge(dut.clk)
        if int(dut.ready.value):
            break


async def step(dut, q, k, vs, g, beta) -> tuple[list[np.ndarray], int]:
    for _ in range(D + 64):
        if int(dut.ready.value):
            break
        await RisingEdge(dut.clk)
    dut.q_flat.value = pack_signed(q, QK_W)
    dut.k_flat.value = pack_signed(k, QK_W)
    packed_v = 0
    for h, v in enumerate(vs):
        packed_v |= pack_signed(v, V_W) << (h * D * V_W)
    dut.v_flat.value = packed_v
    dut.g.value = int(g)
    dut.beta.value = int(beta)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(1, unit="ns")
    cycles = 1
    for _ in range(D + 64):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            raw = int(dut.o_flat.value)
            outs = [unpack_signed(raw >> (h * D * O_W), D, O_W) for h in range(N_V)]
            return outs, cycles
    raise AssertionError("gqa timeout")


@cocotb.test()
async def test_gqa3_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(7)
    S = [np.zeros((D, D), dtype=np.int64) for _ in range(N_V)]
    n_ok = 0
    cycs = []
    for _ in range(8):
        q = rng.integers(-128, 128, size=D, dtype=np.int64)
        k = rng.integers(-128, 128, size=D, dtype=np.int64)
        vs = [rng.integers(-128, 128, size=D, dtype=np.int64) for _ in range(N_V)]
        g = int(rng.integers(0, 256))
        beta = int(rng.integers(0, 256))
        y, cyc = await step(dut, q, k, vs, g, beta)
        cycs.append(cyc)
        for h in range(N_V):
            S[h], y_ref = gated_delta_step(S[h], q, k, vs[h], g, beta)
            np.testing.assert_array_equal(y[h], y_ref)
        n_ok += 1
    rec = {
        "gate": "gqa3 == 3x gated_delta_step, shared q/k/g/beta",
        "status": "PASS",
        "dut": "rtl/gated_delta_gqa3.v",
        "d": D,
        "n_v": N_V,
        "tokens": n_ok,
        "cycles_per_token": int(round(sum(cycs) / len(cycs))),
        "cycles_min": min(cycs),
        "cycles_max": max(cycs),
        "note": "3 V-heads overlap; layer cycles equal one fused PE, not 3x.",
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / "delta_gqa3_equiv.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"gqa3 PASS {n_ok} cyc={rec['cycles_per_token']}")
