import math
from pathlib import Path

import numpy as np
import cocotb
from cocotb.triggers import Timer

ROOT = Path(__file__).resolve().parents[1]
N = 128
IN_W = 8
W_W = 4
ACC_W = IN_W + W_W + int(math.ceil(math.log2(N)))


def pack_signed(xs: np.ndarray, width: int) -> int:
    mask = (1 << width) - 1
    v = 0
    for i, x in enumerate(xs.tolist()):
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


def activation_vectors(n: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(1)
    vecs: list[tuple[str, np.ndarray]] = [
        ("zeros", np.zeros(n, dtype=np.int32)),
        ("ones", np.ones(n, dtype=np.int32)),
        ("max", np.full(n, 127, dtype=np.int32)),
        ("min", np.full(n, -128, dtype=np.int32)),
        ("alt", np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int32)),
    ]
    x_path = ROOT / "artifacts" / "ref_x_int8.npy"
    if x_path.exists():
        vecs.append(("ref_x", np.load(x_path).astype(np.int32)))
    for i in range(n):
        e = np.zeros(n, dtype=np.int32)
        e[i] = 1
        vecs.append((f"e{i}", e))
        e = np.zeros(n, dtype=np.int32)
        e[i] = -1
        vecs.append((f"neg_e{i}", e))
    for s in range(8):
        vecs.append((f"rand{s}", rng.integers(-128, 128, size=n, dtype=np.int32)))
    return vecs


async def drive_and_check(dut, x: np.ndarray, y_ref: np.ndarray, name: str) -> None:
    dut.x_flat.value = pack_signed(x, IN_W)
    await Timer(1, unit="ns")
    y = unpack_signed(int(dut.y_flat.value), N, ACC_W)
    if not np.array_equal(y, y_ref):
        mism = np.where(y != y_ref)[0]
        i = int(mism[0])
        raise AssertionError(
            f"{name}: mismatch at row {i}: rtl={int(y[i])} numpy={int(y_ref[i])} "
            f"({len(mism)} rows differ)"
        )


@cocotb.test()
async def test_bit_exact_vs_numpy_quantized_matvec(dut):
    w = np.load(ROOT / "artifacts" / "tile128_int4.npy").astype(np.int32)
    assert w.shape == (N, N)
    n_ok = 0
    for name, x in activation_vectors(N):
        y_ref = w @ x
        await drive_and_check(dut, x, y_ref, name)
        n_ok += 1
    dut._log.info(f"bit-exact: {n_ok} activation vectors matched numpy W_int @ x")
