"""Bit-exact cocotb vs numpy for spatial FFN columns (serial CSD, tap, fetch)."""

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

from pe_xbar.weights import bit_serial_matvec, load_w  # noqa: E402

N = int(os.environ.get("FFN_TILE_N", "8"))
IN_W = 8
W_W = 4
ACC_W = IN_W + W_W + int(math.ceil(math.log2(max(N, 2))))
DUT = os.environ.get("XBAR_DUT", "ffn_col_serial")


def pack_signed(xs: np.ndarray, width: int) -> int:
    mask = (1 << width) - 1
    v = 0
    for i, x in enumerate(np.asarray(xs, dtype=np.int64).tolist()):
        v |= (int(x) & mask) << (i * width)
    return v


def unpack_signed(val: int, n: int, width: int) -> np.ndarray:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        raw = (val >> (i * width)) & mask
        out[i] = raw - (1 << width) if raw & sign else raw
    return out


def vectors(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(2)
    vecs = [
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=np.int32),
        np.full(n, 127, dtype=np.int32),
        np.full(n, -128, dtype=np.int32),
        np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int32),
    ]
    for i in range(n):
        e = np.zeros(n, dtype=np.int32)
        e[i] = 1
        vecs.append(e)
        e = np.zeros(n, dtype=np.int32)
        e[i] = -1
        vecs.append(e)
    for _ in range(8):
        vecs.append(rng.integers(-128, 128, size=n, dtype=np.int32))
    return vecs


async def reset(dut) -> None:
    dut.en.value = 0
    dut.x_flat.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def step_en(dut, x: np.ndarray) -> tuple[np.ndarray, int]:
    dut.x_flat.value = pack_signed(x, IN_W)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(1, unit="ns")
    cycles = 1
    for _ in range(max(4096, N * N + 16)):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            y = unpack_signed(int(dut.y_flat.value), N, ACC_W)
            return y, cycles
    raise AssertionError(f"{DUT}: timed out waiting for done")


def write_equiv(dut_name: str, n_ok: int, cycles: list[int]) -> None:
    rec = {
        "gate": "rtl_y == numpy(W_int @ x_int)",
        "status": "PASS",
        "dut": f"rtl/{DUT}.v",
        "name": dut_name,
        "n": N,
        "bits": W_W,
        "vectors": n_ok,
        "cycles_per_token": int(round(sum(cycles) / len(cycles))) if cycles else 1,
        "cycles_min": min(cycles) if cycles else 1,
        "cycles_max": max(cycles) if cycles else 1,
        "weights": f"artifacts/tile{N}_int4_xbar.npy" if N != 8 else "artifacts/tile8_int4_xbar.npy",
        "simulator": "Verilator + cocotb",
    }
    out = ROOT / "artifacts" / f"{dut_name}_equiv.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


@cocotb.test()
async def test_xbar_bit_exact(dut):
    w = load_w(N).astype(np.int32)
    assert w.shape == (N, N), w.shape
    vecs = vectors(N)

    clocked_reg = DUT in ("ffn_rom_tap_reg", "ffn_tile_reg")
    if DUT == "ffn_rom_tap" and not clocked_reg:
        n_ok = 0
        for x in vecs:
            dut.x_flat.value = pack_signed(x, IN_W)
            await Timer(1, unit="ns")
            y = unpack_signed(int(dut.y_flat.value), N, ACC_W)
            np.testing.assert_array_equal(y, w @ x)
            n_ok += 1
        rec = write_equiv(f"ffn_rom_tap_{N}x{N}_b4", n_ok, [1] * n_ok)
        dut._log.info(f"rom_tap bit-exact: {n_ok} vectors, cycles={rec['cycles_per_token']}")
        return

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    if clocked_reg:
        n_ok = 0
        for x in vecs:
            dut.x_flat.value = pack_signed(x, IN_W)
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            y = unpack_signed(int(dut.y_flat.value), N, ACC_W)
            np.testing.assert_array_equal(y, w @ x)
            n_ok += 1
        name = f"{DUT}_{N}x{N}_b4"
        rec = write_equiv(name, n_ok, [1] * n_ok)
        dut._log.info(f"{DUT} registered bit-exact: {n_ok} vectors")
        return

    await reset(dut)
    n_ok = 0
    cycle_counts: list[int] = []
    for x in vecs:
        y, cyc = await step_en(dut, x)
        y_ref = w @ x
        if DUT == "ffn_col_serial":
            y_ser = bit_serial_matvec(w, x)
            np.testing.assert_array_equal(y_ser, y_ref)
        np.testing.assert_array_equal(y, y_ref)
        n_ok += 1
        cycle_counts.append(cyc)

    name = f"{DUT}_{N}x{N}_b4"
    rec = write_equiv(name, n_ok, cycle_counts)
    dut._log.info(f"{DUT} bit-exact: {n_ok} vectors, cycles/token={rec['cycles_per_token']}")
