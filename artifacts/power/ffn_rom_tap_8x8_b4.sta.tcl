set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -power mW
read_liberty /Users/toprakgundogdu/.volare/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog /Users/toprakgundogdu/qwen-ffn-rom/openlane/runs/xbar_rom_tap/final/nl/ffn_rom_tap_reg.nl.v
link_design ffn_rom_tap_reg
read_spef /Users/toprakgundogdu/qwen-ffn-rom/openlane/runs/xbar_rom_tap/final/spef/nom/ffn_rom_tap_reg.nom.spef
create_clock -name clk -period 50.0 [get_ports clk]
set_power_activity -global -activity 0.126942 -duty 0.5
report_checks -path_delay max -digits 4 > /Users/toprakgundogdu/qwen-ffn-rom/artifacts/power/ffn_rom_tap_8x8_b4.power.rpt.timing
report_power -digits 6 > /Users/toprakgundogdu/qwen-ffn-rom/artifacts/power/ffn_rom_tap_8x8_b4.power.rpt
