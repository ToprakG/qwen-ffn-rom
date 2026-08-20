SIM ?= verilator
TOPLEVEL_LANG ?= verilog

PWD := $(shell pwd)
ROOT := $(abspath $(PWD)/..)

VERILOG_SOURCES += $(ROOT)/rtl/adder.v
COCOTB_TEST_MODULES := test_adder
TOPLEVEL := adder

ifeq ($(SIM),verilator)
EXTRA_ARGS += --timescale 1ns/1ps -Wno-fatal
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
