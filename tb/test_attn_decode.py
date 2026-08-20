"""Count clocks for D-parallel attention decode (no softmax). Fit cycles vs S."""

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

from quant.layer_int import attn_decode_int  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

D = 4
O_W = 24


async def reset(dut) -> None:
    dut.en.value = 0
    dut.seq_len.value = 1
    dut.q_flat.value = 0
    dut.k_flat.value = 0
    dut.v_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_attn_decode_cycles(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(2)
    recs = []
    for S in (8, 16, 32):
        q = rng.integers(-64, 64, size=D, dtype=np.int64)
        K = rng.integers(-64, 64, size=(S, D), dtype=np.int64)
        V = rng.integers(-64, 64, size=(S, D), dtype=np.int64)
        kpack = 0
        vpack = 0
        for t in range(S):
            kpack |= pack_signed(K[t], 8) << (t * D * 8)
            vpack |= pack_signed(V[t], 8) << (t * D * 8)
        dut.q_flat.value = pack_signed(q, 8)
        dut.k_flat.value = kpack
        dut.v_flat.value = vpack
        dut.seq_len.value = S
        dut.en.value = 1
        await Timer(1, unit="ns")
        await RisingEdge(dut.clk)
        dut.en.value = 0
        cycles = 1
        for _ in range(8 * S + 64):
            await RisingEdge(dut.clk)
            cycles += 1
            await Timer(1, unit="ns")
            if int(dut.done.value):
                break
        else:
            raise AssertionError(f"attn timeout S={S}")
        o = unpack_signed(int(dut.o_flat.value), D, O_W)
        np.testing.assert_array_equal(o, attn_decode_int(q, K, V))
        recs.append({"S": S, "cycles": cycles, "two_S": 2 * S, "overhead": cycles - 2 * S})

    Ss = np.array([r["S"] for r in recs], dtype=np.float64)
    Cs = np.array([r["cycles"] for r in recs], dtype=np.float64)
    b, a = np.polyfit(Ss, Cs, 1)
    out = {
        "gate": "attn_decode == (K@q)>>8 then (V.T@scores)>>8 (no softmax)",
        "status": "PASS",
        "dut": "rtl/attn_decode.v",
        "d": D,
        "measured": recs,
        "fit": {
            "intercept": float(a),
            "slope_per_S": float(b),
            "formula": "cycles = intercept + slope * S",
        },
        "extrapolated_one_head": {
            "4096": int(round(a + b * 4096)),
            "32768": int(round(a + b * 32768)),
        },
        "note": (
            "D-wide inner product, one cache row per clock. Softmax is not in RTL; "
            "serial softmax would add about S clocks more per head."
        ),
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / "attn_decode_cycles.json").write_text(json.dumps(out, indent=2) + "\n")
    dut._log.info(f"attn measured {recs} fit slope={b:.3f} intercept={a:.3f}")
