# Vivado out-of-context synth for the BRAM Gated DeltaNet PE.
# Not run in this repo (no Vivado on the build Mac). After install:
#   vivado -mode batch -source vivado/synth.tcl
# Optional: vivado -mode batch -source vivado/synth.tcl -tclargs gated_delta_d128_bram xc7a200tffg1156-2
set top    [expr {[llength $argv] > 0 ? [lindex $argv 0] : "gated_delta_d4_bram"}]
set part   [expr {[llength $argv] > 1 ? [lindex $argv 1] : "xc7a200tffg1156-2"}]
set root   [file normalize [file join [file dirname [info script]] ..]]
set outdir [file join $root artifacts vivado $top]
file mkdir $outdir

set rtl_common [list \
  [file join $root rtl delta_s_col_ram.v] \
  [file join $root rtl gated_delta_bram.v] \
]

switch $top {
  gated_delta_d4_bram   { set extra [list [file join $root rtl gated_delta_d4_bram.v]] }
  gated_delta_d16_bram  { set extra [list [file join $root rtl gated_delta_d16_bram.v]] }
  gated_delta_d128_bram { set extra [list [file join $root rtl gated_delta_d128_bram.v]] }
  qwen08b_farm_d4 {
    set extra [list \
      [file join $root rtl qwen08b_delta_farm.v] \
      [file join $root rtl qwen08b_farm_d4.v] \
    ]
  }
  qwen08b_heads16_d4 {
    set extra [list \
      [file join $root rtl qwen08b_delta_farm.v] \
      [file join $root rtl qwen08b_heads16_d4.v] \
    ]
  }
  fpga_top {
    set extra [list \
      [file join $root rtl qwen08b_delta_farm.v] \
      [file join $root rtl qwen08b_heads16_d4.v] \
      [file join $root rtl uart_rx.v] \
      [file join $root rtl uart_tx.v] \
      [file join $root rtl fpga_top.v] \
    ]
  }
  default { error "unknown top $top" }
}

create_project -in_memory -part $part
foreach f [concat $rtl_common $extra] {
  read_verilog -sv $f
}
synth_design -top $top -mode out_of_context
opt_design
report_utilization -file [file join $outdir util.rpt]
report_timing_summary -file [file join $outdir timing.rpt]
# Fmax from WNS at a 5 ns constraint would go here after create_clock.
puts "wrote $outdir"
