# Focused OpenSTA: D=128 mixer post-synth netlist, SS corner, rst_n false-path.
# Not post-route. Interconnect is ideal (no SPEF).
set LIB $::env(STA_LIB)
set NL  $::env(STA_NL)
set TOP $::env(STA_TOP)
set PER $::env(STA_PERIOD_NS)

read_liberty $LIB
read_verilog $NL
link_design $TOP

create_clock [get_ports clk] -name clk -period $PER
set_false_path -from [get_ports rst_n]
set_output_delay 0 -clock clk [all_outputs]

puts "=== report_wns ==="
report_wns
puts "=== report_tns ==="
report_tns
puts "=== worst setup ==="
report_checks -path_delay max -format full -group_count 1 -fields {slew cap fanout}
puts "=== done ==="
