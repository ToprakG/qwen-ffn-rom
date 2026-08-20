# AWS F2 trial (mixer farm, not 35B-A3B)

This is **not** Qwen3.6-35B-A3B in HBM. It is the **same DUT** as `rtl/fpga_top.v`
(16× D=4 DeltaNet farm, 24 layers in time) on F2 **OCL MMIO** instead of UART.

ASIC tok/s and 35B e2e on F2 are different projects. See the chat notes:
full Q4 ~22 GB **does not fit** 16 GiB HBM.

## What you need (you, not this laptop)

1. AWS account, **F2 quota** on `f2.6xlarge` (often a support ticket).
2. Region with F2: `us-east-1` or `us-west-2`. On-demand **~$1.98/hr**.
3. FPGA Developer AMI + **Vivado** (AWS HDK). This repo has **no** Vivado license.
4. Clone HDK: `git clone -b f2 https://github.com/aws/aws-fpga.git`

## Build

Copy `cl_qwen_farm_mmio.v` plus the farm RTL into a CL example
(`cl_hello_world` / `cl_sde` — use whatever the current F2 HDK ships).
Hook `wr_en`/`rd_en`/`addr`/`wr_data`/`rd_data` to the **OCL AXI-Lite**
slave (32-bit registers). File list:

```
f2/cl_qwen_farm_mmio.v
rtl/qwen08b_heads16_d4.v
rtl/qwen08b_delta_farm.v
rtl/gated_delta_bram.v
rtl/gated_delta_d4_bram.v
rtl/delta_s_col_ram.v
```

(Include whatever those modules `include` / instantiate — `scripts/eda_fpga.py`
file lists are the source of truth.)

`./aws_build_dcp_from_cl.sh` → submit DCP → **AFI**. Load on the instance:

```
sudo fpga-load-local-image -S 0 -I agfi-xxxxxxxx
```

## Host

```
python scripts/f2_host.py --dry-run
# after AFI: python scripts/f2_host.py --bar ocl
```

Prints kick→done **cycles** and wall tok/s of the **mixer slice**, not 35B tokens.

## 35B on F2 (later)

HBM 16 GiB + DDR4 64 GiB, Q4 library on DDR, 566 MB hot in HBM, tens–hundreds
of 8×8 taps. Months of CL, not this folder.
