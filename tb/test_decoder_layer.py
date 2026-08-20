"""One decoder layer: rms → DeltaNet D=4 → residual → rms → 8x8 FFN tap → residual."""

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
from quant.layer_int import decoder_layer_int  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

N = 8


async def reset(dut) -> None:
    dut.en.value = 0
    dut.x_flat.value = 0
    dut.g.value = 0
    dut.beta.value = 0
    dut.w_n1_flat.value = 0
    dut.w_n2_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    for _ in range(64):
        await RisingEdge(dut.clk)
        if int(dut.ready.value):
            break


async def step(dut, x, g, beta, w1, w2) -> tuple[np.ndarray, int]:
    for _ in range(64):
        if int(dut.ready.value):
            break
        await RisingEdge(dut.clk)
    dut.x_flat.value = pack_signed(x, 8)
    dut.g.value = int(g)
    dut.beta.value = int(beta)
    dut.w_n1_flat.value = pack_signed(w1, 8)
    dut.w_n2_flat.value = pack_signed(w2, 8)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    cycles = 1
    for _ in range(512):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            y = unpack_signed(int(dut.y_flat.value), N, 8)
            return y, cycles
    raise AssertionError("decoder_layer timeout")


@cocotb.test()
async def test_decoder_layer_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    w_ffn = load_w(N).astype(np.int64)
    rng = np.random.default_rng(3)
    w1 = np.full(N, 64, dtype=np.int64)
    w2 = np.full(N, 64, dtype=np.int64)

    S = np.zeros((4, 4), dtype=np.int64)
    n_ok = 0
    cycs: list[int] = []
    for _ in range(12):
        x = rng.integers(-64, 64, size=N, dtype=np.int64)
        g = int(rng.integers(1, 200))
        b = int(rng.integers(1, 200))
        y, cyc = await step(dut, x, g, b, w1, w2)
        S, y_ref = decoder_layer_int(x, S, g, b, w1, w2, w_ffn)
        np.testing.assert_array_equal(y, y_ref)
        n_ok += 1
        cycs.append(cyc)

    rec = {
        "gate": "decoder_layer == rms1 → gated_delta_step → residual → rms2 → W@h2 → residual",
        "status": "PASS",
        "dut": "rtl/decoder_layer.v",
        "hidden": 8,
        "mixer_d": 4,
        "ffn": "8x8 ffn_rom_tap",
        "tokens": n_ok,
        "cycles_per_layer": int(round(sum(cycs) / len(cycs))),
        "cycles_min": min(cycs),
        "cycles_max": max(cycs),
        "simulator": "Verilator + cocotb",
        "note": "Toy width. Proves the layer flow, not 0.8B/27B dims.",
    }
    (ROOT / "artifacts" / "decoder_layer_equiv.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"decoder_layer PASS {n_ok} tokens cyc={rec['cycles_per_layer']}")
