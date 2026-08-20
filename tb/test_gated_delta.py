"""Bit-exact cocotb vs numpy for the Gated DeltaNet PE (recurrent)."""

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

D = int(os.environ.get("DELTA_D", "4"))
MAC = int(os.environ.get("DELTA_MAC", "1"))
N_LAYERS = int(os.environ.get("DELTA_LAYERS", "1"))
EQUIV_NAME = os.environ.get("DELTA_EQUIV", "")
DUT_RTL = os.environ.get("DELTA_DUT", "")


def pack_signed(xs: np.ndarray, width: int) -> int:
    mask = (1 << width) - 1
    v = 0
    for i, x in enumerate(np.asarray(xs, dtype=np.int64).tolist()):
        v |= (int(x) & mask) << (i * width)
    return v


def unpack_signed(val: int, n: int, width: int) -> np.ndarray:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        raw = (val >> (i * width)) & mask
        out[i] = raw - (1 << width) if raw & sign else raw
    return out


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
    if hasattr(dut, "ready"):
        for _ in range(N_LAYERS * D + 64):
            await RisingEdge(dut.clk)
            if int(dut.ready.value):
                break
        zeros = np.zeros(D, dtype=np.int64)
        await step_token(dut, zeros, zeros, zeros, 0, 0)


async def step_token(dut, q, k, v, g, beta) -> tuple[np.ndarray, int]:
    if hasattr(dut, "ready"):
        for _ in range(N_LAYERS * D + 64):
            if int(dut.ready.value):
                break
            await RisingEdge(dut.clk)
    dut.q_flat.value = pack_signed(q, QK_W)
    dut.k_flat.value = pack_signed(k, QK_W)
    dut.v_flat.value = pack_signed(v, V_W)
    dut.g.value = int(g)
    dut.beta.value = int(beta)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(1, unit="ns")
    cycles = 1
    limit = max(256, 16 * D * D + 32, N_LAYERS * (16 * D + 64) + 128)
    for _ in range(limit):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            y = unpack_signed(int(dut.o_flat.value), D, O_W)
            return y, cycles
    raise AssertionError("timed out waiting for done")


def rand_vec(rng: np.random.Generator, lo: int, hi: int) -> np.ndarray:
    return rng.integers(lo, hi, size=D, dtype=np.int64)


@cocotb.test()
async def test_gated_delta_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(4)
    families: list[list[tuple]] = []

    zeros = np.zeros(D, dtype=np.int64)
    mid = np.full(D, 64, dtype=np.int64)
    families.append([(zeros, zeros, zeros, 0, 0)])
    families.append([(mid, mid, mid, 255, 255)])
    families.append([(np.full(D, 127), np.full(D, -128), np.full(D, 127), 128, 200)])

    walk = []
    for i in range(D):
        e = np.zeros(D, dtype=np.int64)
        e[i] = 64
        walk.append((e, e, e, 200, 180))
    families.append(walk)

    seq = []
    n_rand = 8 if D > 4 else 16
    for _ in range(n_rand):
        seq.append(
            (
                rand_vec(rng, -128, 128),
                rand_vec(rng, -128, 128),
                rand_vec(rng, -128, 128),
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
            )
        )
    families.append(seq)

    hold = [
        (rand_vec(rng, -64, 64), rand_vec(rng, -64, 64), rand_vec(rng, -64, 64), 255, 0)
        for _ in range(4)
    ]
    families.append(hold)
    rewrite = [
        (rand_vec(rng, -64, 64), rand_vec(rng, -64, 64), rand_vec(rng, -64, 64), 0, 255)
        for _ in range(4)
    ]
    families.append(rewrite)

    n_tokens = 0
    n_seq = 0
    cycle_counts: list[int] = []
    for tokens in families:
        await reset(dut)
        S = np.zeros((D, D), dtype=np.int64)
        n_seq += 1
        for q, k, v, g, beta in tokens:
            y, cyc = await step_token(dut, q, k, v, g, beta)
            S, y_ref = gated_delta_step(S, q, k, v, g, beta)
            if not np.array_equal(y, y_ref):
                dut._log.warning(
                    f"mismatch token={n_tokens} q={q.tolist()} k={k.tolist()} "
                    f"v={v.tolist()} g={g} beta={beta} y={y.tolist()} ref={y_ref.tolist()}"
                )
            np.testing.assert_array_equal(y, y_ref)
            n_tokens += 1
            cycle_counts.append(cyc)

    if DUT_RTL:
        dut_name = DUT_RTL
    elif MAC > 1:
        dut_name = "rtl/gated_delta_d16_par.v"
    elif D != 4:
        dut_name = "rtl/gated_delta_d16.v"
    else:
        dut_name = "rtl/gated_delta_step.v"
    rec_cyc = int(round(sum(cycle_counts) / len(cycle_counts)))
    rec = {
        "gate": "rtl_o == numpy gated_delta_step (integer Q0.8)",
        "status": "PASS",
        "dut": dut_name,
        "d": D,
        "qk_w": QK_W,
        "v_w": V_W,
        "o_w": O_W,
        "shift": SHIFT,
        "sequences": n_seq,
        "tokens": n_tokens,
        "cycles_per_token": rec_cyc,
        "cycles_min": min(cycle_counts),
        "cycles_max": max(cycle_counts),
        "mac": MAC,
        "n_layers": N_LAYERS,
        "model_layers_08b": 24,
        "model_layers_27b": 64,
        "cycles_per_layer": rec_cyc if N_LAYERS == 1 else int(round(rec_cyc / N_LAYERS)),
        "cycles_per_model_token": rec_cyc if N_LAYERS > 1 else rec_cyc * 24,
        "simulator": "Verilator + cocotb",
    }
    if EQUIV_NAME:
        out = ROOT / "artifacts" / EQUIV_NAME
    elif MAC > 1:
        out = ROOT / "artifacts" / "delta_dpar16_equiv.json"
    elif D != 4:
        out = ROOT / "artifacts" / "delta_d16_equiv.json"
    else:
        out = ROOT / "artifacts" / "delta_equiv_gate.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(
        f"bit-exact: {n_tokens} tokens in {n_seq} sequences matched numpy; "
        f"cycles/token={rec['cycles_per_token']}"
    )
