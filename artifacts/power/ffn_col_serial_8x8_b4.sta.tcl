set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -power mW
read_liberty /Users/toprakgundogdu/.volare/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog /Users/toprakgundogdu/qwen-ffn-rom/openlane/runs/xbar_col_serial/final/nl/ffn_col_serial.nl.v
link_design ffn_col_serial
read_spef /Users/toprakgundogdu/qwen-ffn-rom/openlane/runs/xbar_col_serial/final/spef/nom/ffn_col_serial.nom.spef
create_clock -name clk -period 50.0 [get_ports clk]
set_power_activity -global -activity 0.070633 -duty 0.5
report_checks -path_delay max -digits 4 > /Users/toprakgundogdu/qwen-ffn-rom/artifacts/power/ffn_col_serial_8x8_b4.power.rpt.timing
report_power -digits 6 > /Users/toprakgundogdu/qwen-ffn-rom/artifacts/power/ffn_col_serial_8x8_b4.power.rpt
