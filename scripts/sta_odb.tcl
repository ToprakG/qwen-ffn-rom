# Post-CTS (pre-route) OpenSTA on the D=128 mixer ODB.
read_liberty $::env(STA_LIB)
read_db $::env(STA_ODB)

create_clock [get_ports clk] -name clk -period $::env(STA_PERIOD_NS)
set_false_path -from [get_ports rst_n]
set_propagated_clock [all_clocks]

set_wire_rc -signal -layer met2
set_wire_rc -clock -layer met5
estimate_parasitics -placement

puts "=== report_wns ==="
report_wns
puts "=== report_tns ==="
report_tns
puts "=== worst setup ==="
report_checks -path_delay max -format full -group_count 1 -fields {slew cap fanout}
puts "=== done ==="
