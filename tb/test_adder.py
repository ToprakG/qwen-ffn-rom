import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


@cocotb.test()
async def test_reset_and_adds(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.a.value = 0
    dut.b.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    assert int(dut.sum.value) == 0

    vectors = [(0, 0, 0), (1, 1, 2), (255, 1, 256), (255, 255, 510)]
    rng = random.Random(0)
    vectors.extend((rng.randrange(256), rng.randrange(256), None) for _ in range(16))

    for a, b, expected in vectors:
        if expected is None:
            expected = a + b
        dut.a.value = a
        dut.b.value = b
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        got = int(dut.sum.value)
        assert got == expected, f"{a}+{b}: expected {expected}, got {got}"
