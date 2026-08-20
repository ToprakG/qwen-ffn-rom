"""Bit-exact cocotb for the hybrid layer stub: delta D=4 + FFN 8x8 on shared x."""

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
from quant.delta_int import O_W, QK_W, SHIFT, V_W, gated_delta_step  # noqa: E402

D = 4
N = 8
IN_W = 8
ACC_W = 15


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
        raw = (val >> (i * width) ) & mask
        out[i] = raw - (1 << width) if raw & sign else raw
    return out


async def reset(dut) -> None:
    dut.en.value = 0
    dut.x_flat.value = 0
    dut.g.value = 0
    dut.beta.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def step_token(dut, x, g, beta) -> tuple[np.ndarray, np.ndarray, int]:
    dut.x_flat.value = pack_signed(x, IN_W)
    dut.g.value = int(g)
    dut.beta.value = int(beta)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(1, unit="ns")
    cycles = 1
    for _ in range(256):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            o = unpack_signed(int(dut.o_delta_flat.value), D, O_W)
            y = unpack_signed(int(dut.y_ffn_flat.value), N, ACC_W)
            return o, y, cycles
    raise AssertionError("timed out waiting for done")


def rand_vec(rng: np.random.Generator, n: int, lo: int, hi: int) -> np.ndarray:
    return rng.integers(lo, hi, size=n, dtype=np.int64)


@cocotb.test()
async def test_hybrid_bit_exact(dut):
    w = load_w(N).astype(np.int32)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rng = np.random.default_rng(6)
    families: list[list[tuple]] = []

    zeros = np.zeros(N, dtype=np.int64)
    mid = np.full(N, 64, dtype=np.int64)
    families.append([(zeros, 0, 0)])
    families.append([(mid, 255, 255)])
    families.append([(np.concatenate([np.full(D, 127), np.full(D, -128)]), 128, 200)])

    walk = []
    for i in range(N):
        e = np.zeros(N, dtype=np.int64)
        e[i] = 64
        walk.append((e, 200, 180))
    families.append(walk)

    seq = []
    for _ in range(16):
        seq.append((rand_vec(rng, N, -128, 128), int(rng.integers(0, 256)), int(rng.integers(0, 256))))
    families.append(seq)

    n_tokens = 0
    n_seq = 0
    cycle_counts: list[int] = []
    for tokens in families:
        await reset(dut)
        S = np.zeros((D, D), dtype=np.int64)
        n_seq += 1
        for x, g, beta in tokens:
            o, y, cyc = await step_token(dut, x, g, beta)
            q = k = v = x[:D]
            S, o_ref = gated_delta_step(S, q, k, v, g, beta)
            y_ref = w @ x.astype(np.int32)
            np.testing.assert_array_equal(o, o_ref)
            np.testing.assert_array_equal(y, y_ref)
            n_tokens += 1
            cycle_counts.append(cyc)

    rec = {
        "gate": "rtl_o == numpy gated_delta_step(x[:4]); rtl_y == numpy(W_int @ x)",
        "status": "PASS",
        "dut": "rtl/hybrid_layer_stub.v",
        "name": "hybrid_layer_d4_ffn8",
        "d": D,
        "n": N,
        "bits": 4,
        "qk_w": QK_W,
        "v_w": V_W,
        "o_w": O_W,
        "shift": SHIFT,
        "sequences": n_seq,
        "tokens": n_tokens,
        "cycles_per_token": int(round(sum(cycle_counts) / len(cycle_counts))),
        "cycles_min": min(cycle_counts),
        "cycles_max": max(cycle_counts),
        "schedule": "ffn_parallel_with_mixer_on_shared_x",
        "qkv": "q=k=v=x[0:3]",
        "weights": "artifacts/tile8_int4_xbar.npy",
        "simulator": "Verilator + cocotb",
        "note": "architecture cycles/token, not FFN per-PE",
    }
    out = ROOT / "artifacts" / "hybrid_layer_equiv.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(
        f"hybrid bit-exact: {n_tokens} tokens in {n_seq} sequences; "
        f"cycles/token={rec['cycles_per_token']}"
    )
