# qwen-ffn-rom

Cycle-accurate RTL for a **Qwen 3.5 / 3.8** dense hybrid decoder: Gated DeltaNet (3:1) + Gated Attention + SwiGLU. FFN weights are treated as **via-programmed ROM** next to 8×8 CSD taps. S and KV stay in SRAM.

This repo is **not** a foundry tapeout, a via-ROM compiler, or an end-to-end 27B chip. Sky130 OpenLane area is **stdcell**. Reticle estimates are a **separate** cell model. Do not mix them.

Target configs: `models/target.json`  
(`Qwen/Qwen3.5-0.8B`, sibling `Qwen/Qwen3.8-27B`).

## Why FFN is ROM

Decode reads almost every FFN weight every token. Off-die DRAM cannot supply that bandwidth at high tok/s. A via bit is read-only and denser than 6T SRAM. Mixer state writes every token, so it is not ROM.

Cell used in `scripts/rom_estimator.py`: **max(18 F², SRAM/8)** at ≥28 nm, **SRAM/4** on FinFET, array fill **0.62**. Scanner field **26×33 mm**.

## Cycles (farm-hidden)

FFN of layer *L* overlaps the mixer of *L+1*. Epilogue = one mixer window (130 clk at D=128).

Measured building blocks (`artifacts/clocks_token.json`):

| Block | Clk | Source |
|---|---|---|
| 8×8 FFN tap | **2** | `tb/test_ffn_tap_cycles.py` |
| DeltaNet D=128 | **130** (D+2) | `gated_delta_d128_fused` |
| Attn decode | **ceil(S/512)+2** | `attn_online` |
| RMSNorm | **2** each | Newton rsqrt |
| SiLU | **0** extra | folded into tap |

At **seq=4096**, farm-hidden token time and taps to hide FFN:

| Model | clk/token | 8×8 taps | 200 MHz *if* those taps and Fmax exist |
|---|---:|---:|---:|
| 0.8B | **2626** | **2647** | 76k tok/s |
| 27B | **6786** | **64276** | 29k tok/s |

200 MHz and 330 MHz are **goals**. 28 nm Fmax is **unmeasured**. One serial tap is ~24 tok/s at 200 MHz (0.8B), not the farm-hidden column.

Regenerate:

```bash
python scripts/clocks_token.py
```

## Layout

| Path | Contents |
|---|---|
| `rtl/` | Verilog. Mixer, attn, RMS, FFN tap/tile, FPGA UART top |
| `rtl/sta/` | OpenLane STA wrappers |
| `tb/` | cocotb / Python equivalence |
| `quant/` | int4/int3/int2, DeltaNet and attn integer refs |
| `scripts/` | clocks, eval, OpenLane ingest, ROM estimator, FPGA host |
| `openlane/` | Sky130 PnR configs and STA JSON |
| `artifacts/` | measured JSON (`eval.json`, `clocks_token.json`, …) |
| `models/` | target architecture |
| `f2/` | AWS F2 MMIO wrapper around the D=4 farm. **No AFI in-tree** |

## Run

Python 3.11+, iverilog or a cocotb-capable sim. Do not commit `.venv/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/eval_bench.py      # ingest artifacts, gates
python scripts/eval_bench.py --sim
python scripts/rom_estimator.py
```

FPGA host (7-series UART top `rtl/fpga_top.v`, 16× D=4 mixer, **not** 27B e2e):

```bash
python scripts/fpga_host.py --dry-run
python scripts/fpga_host.py --port /dev/ttyUSB1
```

Eval contract: `artifacts/eval.schema.json`. Primary metrics: `area_um2`, `tok_s_per_pe`, `nJ_per_token`, `cycles_per_token`. `nJ_per_token` is OpenSTA **vectorless** unless `power_model` says otherwise.

## Status

| Item | State |
|---|---|
| DeltaNet / attn / RMS / FFN tap vs Python | **PASS** in `artifacts/eval.json` |
| Sky130 OpenLane Fmax (D=4 mixer) | **~41 MHz** post-route OpenSTA |
| 7-series FPGA Fmax | **unmeasured** (host assumes 200 MHz) |
| 28 nm / 7 nm STA | **none** |
| Via-ROM bitcell silicon | **none** — estimator only |
| Full 0.8B or 27B netlist | **none** — slices + cycle math |

## License

Add a license before a public clone. Weights stay with their Hugging Face terms (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.8-27B`).
