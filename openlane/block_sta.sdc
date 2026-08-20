# Block STA: do not let async rst_n fanout set Fmax.
create_clock [get_ports clk] -name clk -period $::env(CLOCK_PERIOD)

set input_delay_value [expr $::env(CLOCK_PERIOD) * $::env(IO_DELAY_CONSTRAINT) / 100]
set output_delay_value [expr $::env(CLOCK_PERIOD) * $::env(IO_DELAY_CONSTRAINT) / 100]
set_max_fanout $::env(MAX_FANOUT_CONSTRAINT) [current_design]

set_false_path -from [get_ports rst_n]

set_input_delay $input_delay_value -clock [get_clocks clk] [all_inputs]
set_output_delay $output_delay_value -clock [get_clocks clk] [all_outputs]
# clk must not have input delay
set_input_delay 0 -clock [get_clocks clk] [get_ports clk]
set_false_path -from [get_ports rst_n]

set_clock_uncertainty $::env(CLOCK_UNCERTAINTY_CONSTRAINT) [get_clocks clk]
set_clock_transition $::env(CLOCK_TRANSITION_CONSTRAINT) [get_clocks clk]
set_timing_derate -early [expr 1 - $::env(TIME_DERATING_CONSTRAINT) / 100]
set_timing_derate -late  [expr 1 + $::env(TIME_DERATING_CONSTRAINT) / 100]
