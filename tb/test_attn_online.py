"""Bit-exact fused online-softmax attention vs quant/attn_online_int.py."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.attn_online_int import (  # noqa: E402
    O_W,
    attn_online_gqa_int,
    attn_online_int,
)
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

D = int(os.environ.get("ATTN_D", "8"))
P = int(os.environ.get("ATTN_P", "4"))
N_KV = int(os.environ.get("ATTN_N_KV", "1"))
N_Q_PER = int(os.environ.get("ATTN_N_Q_PER", "1"))
EQUIV_NAME = os.environ.get("ATTN_EQUIV", "attn_online_equiv.json")
DUT_RTL = os.environ.get("ATTN_DUT", "rtl/attn_online_d8_p4.v")


async def reset(dut) -> None:
    dut.en.value = 0
    dut.wr_en.value = 0
    dut.wr_t.value = 0
    dut.wr_k.value = 0
    dut.wr_v.value = 0
    dut.seq_len.value = 1
    dut.q_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_kv(dut, K: np.ndarray, V: np.ndarray) -> None:
    """K,V: (n_kv, S, D) int4 codes."""
    n_kv, s_len, d = K.shape
    for t in range(s_len):
        kpack = 0
        vpack = 0
        for kv in range(n_kv):
            kpack |= pack_signed(K[kv, t], 4) << (kv * d * 4)
            vpack |= pack_signed(V[kv, t], 4) << (kv * d * 4)
        dut.wr_t.value = t
        dut.wr_k.value = kpack
        dut.wr_v.value = vpack
        dut.wr_en.value = 1
        await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)


async def run_once(
    dut, Q: np.ndarray, K: np.ndarray, V: np.ndarray, s_len: int
) -> tuple[np.ndarray, int]:
    n_q = Q.shape[0]
    qpack = 0
    for h in range(n_q):
        qpack |= pack_signed(Q[h], 8) << (h * D * 8)
    dut.q_flat.value = qpack
    dut.seq_len.value = s_len
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    cycles = 1
    for _ in range(8 * s_len + 64):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            break
    else:
        raise AssertionError(f"attn_online timeout S={s_len}")
    o = unpack_signed(int(dut.o_flat.value), n_q * D, O_W).reshape(n_q, D)
    return o, cycles


@cocotb.test()
async def test_attn_online_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    rng = np.random.default_rng(7)
    recs = []
    n_q = N_KV * N_Q_PER
    seqs = (4, 8, 16, 32) if N_KV == 1 else (4, 8, 16)
    for S in seqs:
        Q = rng.integers(-64, 64, size=(n_q, D), dtype=np.int64)
        K = rng.integers(-8, 8, size=(N_KV, S, D), dtype=np.int64)
        V = rng.integers(-8, 8, size=(N_KV, S, D), dtype=np.int64)
        await load_kv(dut, K, V)
        o_rtl, cycles = await run_once(dut, Q, K, V, S)
        if N_KV == 1 and N_Q_PER == 1:
            gold = attn_online_int(Q[0], K[0], V[0], P)
            np.testing.assert_array_equal(o_rtl[0], gold)
        else:
            gold = attn_online_gqa_int(Q, K, V, P, N_Q_PER)
            np.testing.assert_array_equal(o_rtl, gold)
        n_steps = math.ceil(S / P)
        recs.append({
            "S": S,
            "cycles": cycles,
            "n_steps": n_steps,
            "overhead": cycles - n_steps,
        })

    expected_oh = recs[0]["overhead"]
    for r in recs:
        if r["overhead"] != expected_oh:
            raise AssertionError(f"pipe not constant: {recs}")

    out = {
        "gate": "fused online-softmax, P-way KV, int4 cache, == attn_online_int",
        "status": "PASS",
        "dut": DUT_RTL,
        "d": D,
        "p": P,
        "n_kv": N_KV,
        "n_q_per_kv": N_Q_PER,
        "kv_w": 4,
        "measured": recs,
        "formula": f"cycles = ceil(S/{P}) + {expected_oh}",
        "pipe": expected_oh,
        "note": (
            "One pass, no S-length score vector. GQA reuses each KV head across "
            f"{N_Q_PER} Q heads; cycles do not scale with Q or KV head count."
        ),
        "simulator": "Verilator + cocotb",
    }
    (ROOT / "artifacts" / EQUIV_NAME).write_text(json.dumps(out, indent=2) + "\n")
    dut._log.info(f"attn_online PASS {recs} pipe={expected_oh}")
