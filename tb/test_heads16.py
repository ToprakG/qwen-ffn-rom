"""16-head D=4 farm: each head matches numpy gated_delta_step."""

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

from quant.delta_int import O_W, QK_W, V_W, gated_delta_step  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

D = 4
N_HEADS = 16
N_LAYERS = 24


def pack_heads(vecs: list[np.ndarray], width: int) -> int:
    v = 0
    for h, xs in enumerate(vecs):
        v |= pack_signed(xs, width) << (h * D * width)
    return v


async def reset(dut) -> None:
    dut.en.value = 0
    dut.q_flat.value = 0
    dut.k_flat.value = 0
    dut.v_flat.value = 0
    dut.g_flat.value = 0
    dut.beta_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    for _ in range(N_LAYERS * D + 128):
        await RisingEdge(dut.clk)
        if int(dut.ready.value):
            break


async def step_token(dut, qs, ks, vs, gs, betas) -> tuple[list[np.ndarray], int]:
    for _ in range(64):
        if int(dut.ready.value):
            break
        await RisingEdge(dut.clk)
    dut.q_flat.value = pack_heads(qs, QK_W)
    dut.k_flat.value = pack_heads(ks, QK_W)
    dut.v_flat.value = pack_heads(vs, V_W)
    g = 0
    b = 0
    for h in range(N_HEADS):
        g |= (int(gs[h]) & 0xFF) << (h * 8)
        b |= (int(betas[h]) & 0xFF) << (h * 8)
    dut.g_flat.value = g
    dut.beta_flat.value = b
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    cycles = 1
    for _ in range(N_LAYERS * (16 * D + 64) + 256):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            raw = int(dut.o_flat.value)
            outs = [
                unpack_signed(raw >> (h * D * O_W), D, O_W) for h in range(N_HEADS)
            ]
            return outs, cycles
    raise AssertionError("timed out waiting for done")


@cocotb.test()
async def test_heads16_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(4)
    S = [np.zeros((D, D), dtype=np.int64) for _ in range(N_HEADS)]
    n_tokens = 0
    cycle_counts: list[int] = []

    for _ in range(6):
        qs = [rng.integers(-128, 128, size=D, dtype=np.int64) for _ in range(N_HEADS)]
        ks = [rng.integers(-128, 128, size=D, dtype=np.int64) for _ in range(N_HEADS)]
        vs = [rng.integers(-128, 128, size=D, dtype=np.int64) for _ in range(N_HEADS)]
        gs = [int(rng.integers(0, 256)) for _ in range(N_HEADS)]
        bs = [int(rng.integers(0, 256)) for _ in range(N_HEADS)]
        ys, cyc = await step_token(dut, qs, ks, vs, gs, bs)
        refs = []
        for h in range(N_HEADS):
            S[h], y_ref = gated_delta_step(S[h], qs[h], ks[h], vs[h], gs[h], bs[h])
            refs.append(y_ref)
            np.testing.assert_array_equal(ys[h], y_ref)
        n_tokens += 1
        cycle_counts.append(cyc)

    rec = {
        "gate": "16-head D=4 farm == numpy gated_delta_step per head",
        "status": "PASS",
        "dut": "rtl/qwen08b_heads16_d4.v",
        "d": D,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "tokens": n_tokens,
        "cycles_per_token": int(round(sum(cycle_counts) / len(cycle_counts))),
        "cycles_min": min(cycle_counts),
        "cycles_max": max(cycle_counts),
        "cycles_per_model_token": int(round(sum(cycle_counts) / len(cycle_counts))),
        "model_layers_08b": 24,
        "simulator": "Verilator + cocotb",
    }
    out = ROOT / "artifacts" / "qwen08b_heads16_d4_equiv.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(f"heads16 bit-exact {n_tokens} tokens cyc={rec['cycles_per_token']}")
