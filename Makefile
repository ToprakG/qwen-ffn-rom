VENV ?= .venv
OPENLANE := $(VENV)/bin/openlane
PYTHON := $(VENV)/bin/python
export PDK_ROOT ?= $(HOME)/.volare

.PHONY: sim pnr smoke estimate quantize gen-rtl sim-tile equiv sweep-yosys sweep-pnr sim-delta equiv-delta pnr-delta eda-delta sim-delta-d16 equiv-delta-d16 eda-delta-d16 pnr-delta-d16 sim-delta-dpar equiv-delta-dpar eda-delta-dpar pnr-delta-dpar sim-delta-fused equiv-delta-fused eda-delta-fused pnr-delta-fused sim-delta-gqa gen-xbar sim-xbar equiv-xbar eda-xbar pnr-xbar sweep-n sweep-n-pnr power-xbar sim-hybrid equiv-hybrid eda-hybrid pnr-hybrid floorplan eval eval-sim eval-strict eval-promote sim-delta-bram sim-farm sim-heads16 sim-fpga-top eda-fpga sim-ffn-tap sim-ffn-swiglu sim-attn sim-attn-online equiv-attn-online sim-rmsnorm-fast equiv-norm sim-decoder-layer sim-qwen-layer eda-qwen-layer ingest-qwen-layer pnr-qwen-layer honest-tok clocks-token eda-sky130-blocks pnr-sky130-blocks clean

sim:
	. $(VENV)/bin/activate && $(MAKE) -C tb

pnr:
	$(OPENLANE) --docker-no-tty --dockerized --run-tag adder_e2e --overwrite openlane/config.json

smoke:
	$(OPENLANE) --docker-no-tty --dockerized --smoke-test

estimate:
	$(PYTHON) scripts/rom_estimator.py

quantize:
	$(PYTHON) scripts/quantize_ffn.py

gen-rtl:
	$(PYTHON) scripts/gen_ffn_tile.py

sim-tile:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.tile

equiv: sim-tile
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.small FFN_TILE_N=8 FFN_TILE_BITS=4

sweep-yosys:
	$(PYTHON) scripts/sky130_sweep.py --tiles 4,8,16 --bits 2,3,4

sweep-pnr:
	$(PYTHON) scripts/sky130_sweep.py --tiles 4,8 --bits 2,3,4 --pnr --pnr-max-n 8

sim-delta:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta

equiv-delta: sim-delta

pnr-delta:
	$(OPENLANE) --docker-no-tty --dockerized --run-tag gated_delta_d4 --overwrite openlane/gated_delta_d4.json

eda-delta:
	$(PYTHON) scripts/eda_delta.py --pnr

sim-delta-d16:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_d16

equiv-delta-d16: sim-delta-d16

pnr-delta-d16:
	$(OPENLANE) --docker-no-tty --dockerized --run-tag gated_delta_d16 --overwrite openlane/gated_delta_d16.json

eda-delta-d16:
	$(PYTHON) scripts/eda_delta.py --d 16 --pnr

sim-delta-dpar:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_dpar

equiv-delta-dpar: sim-delta-dpar

pnr-delta-dpar:
	$(OPENLANE) --docker-no-tty --dockerized --run-tag gated_delta_d16_par --overwrite openlane/gated_delta_d16_par.json

eda-delta-dpar:
	$(PYTHON) scripts/eda_delta.py --d 16 --mac 16 --pnr

sim-delta-fused:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_fused D=16
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_fused D=128

equiv-delta-fused: sim-delta-fused

eda-delta-fused:
	$(PYTHON) scripts/eda_delta.py --fused --d 16

pnr-delta-fused:
	$(PYTHON) scripts/eda_delta.py --fused --d 16 --pnr

sim-delta-gqa:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_gqa

floorplan:
	$(PYTHON) scripts/floorplan_08b.py

gen-xbar:
	$(PYTHON) pe_xbar/emit.py

sim-xbar:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_col_serial
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_rom_tap
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_rom_fetch

equiv-xbar: sim-xbar

pnr-xbar:
	$(PYTHON) scripts/eda_xbar.py --pnr

eda-xbar:
	$(PYTHON) scripts/eda_xbar.py --pnr

sweep-n:
	$(PYTHON) scripts/xbar_n_sweep.py --equiv

sweep-n-pnr:
	$(PYTHON) scripts/xbar_n_sweep.py --equiv --pnr --pnr-max-n 16

power-xbar:
	$(PYTHON) scripts/activity_power.py

sim-hybrid:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.hybrid

equiv-hybrid: sim-hybrid

pnr-hybrid:
	$(OPENLANE) --docker-no-tty --dockerized --run-tag hybrid_layer_stub --overwrite openlane/hybrid_layer_stub.json

eda-hybrid:
	$(PYTHON) scripts/eda_hybrid.py --pnr

sim-delta-bram:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_bram D=4
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_bram D=16
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.delta_bram D=128

sim-farm:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.farm

sim-heads16:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.heads16

sim-fpga-top:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.fpga_top

eda-fpga:
	$(PYTHON) scripts/eda_fpga.py

sim-ffn-tap:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.ffn_tap

sim-ffn-swiglu:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.ffn_swiglu

sim-attn:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.attn

sim-attn-online:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.attn_online DUT=pe
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.attn_online DUT=gqa
	$(PYTHON) scripts/attn_int4_quality.py

equiv-attn-online: sim-attn-online

sim-rmsnorm-fast:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.rmsnorm_fast
	$(PYTHON) scripts/rsqrt_quality.py

equiv-norm: sim-rmsnorm-fast sim-ffn-swiglu

sim-decoder-layer:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.decoder_layer

sim-qwen-layer:
	. $(VENV)/bin/activate && $(MAKE) -C tb -f Makefile.qwen_layer

eda-qwen-layer:
	$(PYTHON) scripts/eda_qwen_layer.py

ingest-qwen-layer:
	$(PYTHON) scripts/eda_qwen_layer.py --ingest

pnr-qwen-layer:
	$(PYTHON) scripts/eda_qwen_layer.py --pnr

eda-sky130-blocks:
	$(PYTHON) scripts/eda_sky130_blocks.py

pnr-sky130-blocks:
	$(PYTHON) scripts/eda_sky130_blocks.py --pnr

honest-tok: sim-ffn-tap sim-attn sim-decoder-layer
	$(PYTHON) scripts/honest_tok_s.py

clocks-token:
	$(PYTHON) scripts/clocks_token.py
	MPLCONFIGDIR=$(CURDIR)/artifacts/.mpl $(PYTHON) scripts/draw_maps.py

eval:
	$(PYTHON) scripts/eval_bench.py

eval-sim:
	$(PYTHON) scripts/eval_bench.py --sim

eval-strict:
	$(PYTHON) scripts/eval_bench.py --strict

eval-promote:
	$(PYTHON) scripts/eval_bench.py --promote

clean:
	$(MAKE) -C tb clean
	$(MAKE) -C tb -f Makefile.tile clean
	$(MAKE) -C tb -f Makefile.delta clean
	$(MAKE) -C tb -f Makefile.delta_d16 clean
	$(MAKE) -C tb -f Makefile.delta_dpar clean
	$(MAKE) -C tb -f Makefile.delta_fused D=16 clean
	$(MAKE) -C tb -f Makefile.delta_fused D=128 clean
	$(MAKE) -C tb -f Makefile.attn_online DUT=pe clean
	$(MAKE) -C tb -f Makefile.attn_online DUT=gqa clean
	$(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_col_serial clean
	$(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_rom_tap clean
	$(MAKE) -C tb -f Makefile.xbar XBAR_DUT=ffn_rom_fetch clean
	$(MAKE) -C tb -f Makefile.hybrid clean
	$(MAKE) -C tb -f Makefile.delta_bram D=4 clean
	$(MAKE) -C tb -f Makefile.delta_bram D=16 clean
	$(MAKE) -C tb -f Makefile.delta_bram D=128 clean
	$(MAKE) -C tb -f Makefile.farm clean
	$(MAKE) -C tb -f Makefile.heads16 clean
	$(MAKE) -C tb -f Makefile.fpga_top clean
	$(MAKE) -C tb -f Makefile.ffn_tap clean
	$(MAKE) -C tb -f Makefile.attn clean
	$(MAKE) -C tb -f Makefile.decoder_layer clean
	$(MAKE) -C tb -f Makefile.qwen_layer clean
	rm -rf openlane/runs runs
