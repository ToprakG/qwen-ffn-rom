"""UART loop around fpga_top: one token in, check o and cycle count."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.delta_int import O_W, gated_delta_step  # noqa: E402

DIV = 100_000_000 // 3_000_000  # matches fpga_top default
D = 4
N_HEADS = 16
N_LAYERS = 24


async def bits_byte(dut, byte: int, div: int) -> None:
    dut.uart_rx.value = 0
    await ClockCycles(dut.clk, div)
    for i in range(8):
        dut.uart_rx.value = (byte >> i) & 1
        await ClockCycles(dut.clk, div)
    dut.uart_rx.value = 1
    await ClockCycles(dut.clk, div)


async def rx_byte(dut, div: int, timeout: int = 50_000) -> int:
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if int(dut.uart_tx.value) == 0:
            break
    else:
        raise AssertionError("no start bit")
    await ClockCycles(dut.clk, div // 2)
    v = 0
    for i in range(8):
        await ClockCycles(dut.clk, div)
        v |= int(dut.uart_tx.value) << i
    await ClockCycles(dut.clk, div)
    return v


@cocotb.test()
async def test_fpga_top_uart(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.uart_rx.value = 1
    dut.cpu_reset.value = 1
    await ClockCycles(dut.clk, 8)
    dut.cpu_reset.value = 0
    for _ in range(N_LAYERS * D + 256):
        await RisingEdge(dut.clk)
        if int(dut.led_ready.value):
            break

    rng = np.random.default_rng(1)
    qs = [rng.integers(-64, 64, size=D, dtype=np.int64) for _ in range(N_HEADS)]
    ks = [rng.integers(-64, 64, size=D, dtype=np.int64) for _ in range(N_HEADS)]
    vs = [rng.integers(-64, 64, size=D, dtype=np.int64) for _ in range(N_HEADS)]
    gs = [int(rng.integers(1, 200)) for _ in range(N_HEADS)]
    bs = [int(rng.integers(1, 200)) for _ in range(N_HEADS)]

    payload = bytearray()
    for vecs in (qs, ks, vs):
        for h in range(N_HEADS):
            payload.extend(int(x) & 0xFF for x in vecs[h].tolist())
    payload.extend(gs)
    payload.extend(bs)
    assert len(payload) == 224

    await bits_byte(dut, 0xA5, DIV)
    for b in payload:
        await bits_byte(dut, b, DIV)

    magic = await rx_byte(dut, DIV)
    assert magic == 0x5A, f"bad magic {magic:#x}"
    rsp = bytearray()
    for _ in range(196):
        rsp.append(await rx_byte(dut, DIV))

    refs = []
    for h in range(N_HEADS):
        S = np.zeros((D, D), dtype=np.int64)
        _, y = gated_delta_step(S, qs[h], ks[h], vs[h], gs[h], bs[h])
        refs.append(y)

    for h in range(N_HEADS):
        for i in range(D):
            off = (h * D + i) * 3
            raw = rsp[off] | (rsp[off + 1] << 8) | (rsp[off + 2] << 16)
            if raw & 0x800000:
                raw -= 1 << 24
            np.testing.assert_equal(raw, int(refs[h][i]))

    cyc = rsp[192] | (rsp[193] << 8) | (rsp[194] << 16) | (rsp[195] << 24)
    dut._log.info(f"uart token ok cycles={cyc}")
    assert 24 * 20 < cyc < 24 * 80
