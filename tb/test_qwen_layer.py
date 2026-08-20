"""Complete qwen_layer: rms → 4×(DeltaNet|attn) → residual → rms → 16×16 FFN → residual."""

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

from pe_xbar.weights import parse_w_from_rtl  # noqa: E402
from quant.layer_int import qwen_layer_int  # noqa: E402
from test_gated_delta import pack_signed, unpack_signed  # noqa: E402

H = 16
D = 4
HEADS = 4
S_MAX = 8


async def reset(dut) -> None:
    dut.en.value = 0
    dut.use_attn.value = 0
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


async def step(dut, x, g, beta, w1, w2, use_attn: bool) -> tuple[np.ndarray, int]:
    for _ in range(64):
        if int(dut.ready.value):
            break
        await RisingEdge(dut.clk)
    dut.x_flat.value = pack_signed(x, 8)
    dut.g.value = int(g)
    dut.beta.value = int(beta)
    dut.w_n1_flat.value = pack_signed(w1, 8)
    dut.w_n2_flat.value = pack_signed(w2, 8)
    dut.use_attn.value = int(use_attn)
    dut.en.value = 1
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 0
    cycles = 1
    for _ in range(4096):
        await RisingEdge(dut.clk)
        cycles += 1
        await Timer(1, unit="ns")
        if int(dut.done.value):
            y = unpack_signed(int(dut.y_flat.value), H, 8)
            return y, cycles
    raise AssertionError("qwen_layer timeout")


@cocotb.test()
async def test_qwen_layer_bit_exact(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    w_ffn = parse_w_from_rtl(ROOT / "rtl" / "ffn_tile_16x16_b4_reg.v", n=H).astype(np.int64)
    rng = np.random.default_rng(5)
    w1 = np.full(H, 64, dtype=np.int64)
    w2 = np.full(H, 64, dtype=np.int64)

    rec_cyc = {"delta": [], "attn": []}
    n_ok = 0
    for use_attn, tag in ((False, "delta"), (True, "attn")):
        mix_S = [np.zeros((D, D), dtype=np.int64) for _ in range(HEADS)]
        kv = [(np.zeros((0, D), dtype=np.int64), np.zeros((0, D), dtype=np.int64)) for _ in range(HEADS)]
        for _ in range(6):
            x = rng.integers(-64, 64, size=H, dtype=np.int64)
            g = int(rng.integers(1, 200))
            b = int(rng.integers(1, 200))
            y, cyc = await step(dut, x, g, b, w1, w2, use_attn)
            mix_S, kv, y_ref = qwen_layer_int(
                x, mix_S, kv, use_attn, g, b, w1, w2, w_ffn, d=D, heads=HEADS, s_max=S_MAX
            )
            np.testing.assert_array_equal(y, y_ref)
            rec_cyc[tag].append(cyc)
            n_ok += 1

    rec = {
        "gate": "qwen_layer == rms1 → 4×(DeltaNet|attn) → residual → rms2 → 16×16 FFN → residual",
        "status": "PASS",
        "dut": "rtl/qwen_layer.v",
        "hidden": H,
        "heads": HEADS,
        "mixer_d": D,
        "s_max": S_MAX,
        "ffn": "ffn_tile 16x16 b4 from Qwen3.5-0.8B down_proj",
        "tokens": n_ok,
        "cycles_delta_layer": int(round(sum(rec_cyc["delta"]) / len(rec_cyc["delta"]))),
        "cycles_attn_layer_mean": int(round(sum(rec_cyc["attn"]) / len(rec_cyc["attn"]))),
        "cycles_attn_by_token": rec_cyc["attn"],
        "cycles_delta_by_token": rec_cyc["delta"],
        "simulator": "Verilator + cocotb",
        "note": (
            "Complete layer, not the H=8 toy. H=16 is 4 signed-off D=4 PEs + real 0.8B 16×16 "
            "tile. 0.8B/27B clocks/token use this layer's RMS/FFN handshake plus measured "
            "D=128 mixer and attn ceil(S/512)+2, not these DUT mixer clocks."
        ),
    }
    (ROOT / "artifacts" / "qwen_layer_equiv.json").write_text(json.dumps(rec, indent=2) + "\n")
    dut._log.info(
        f"qwen_layer PASS {n_ok}  delta={rec['cycles_delta_layer']}  "
        f"attn_mean={rec['cycles_attn_layer_mean']}"
    )
