// One S column: DEPTH signed S_W words. D of these bank a row-parallel PE.
// Simple dual-port so Yosys/Vivado infer block RAM. Do not loop-clear in
// an always block — that becomes flops. FPGA BRAM powers up 0; sim uses initial.
`timescale 1ns / 1ps

module delta_s_col_ram #(
  parameter integer DEPTH  = 16,
  parameter integer S_W    = 16,
  parameter integer ADDR_W = $clog2(DEPTH)
) (
  input  wire                     clk,
  input  wire                     we,
  input  wire [ADDR_W-1:0]        waddr,
  input  wire signed [S_W-1:0]    wdata,
  input  wire [ADDR_W-1:0]        raddr,
  output reg signed [S_W-1:0]     rdata
);
  (* ram_style = "block" *)
  (* ramstyle = "M20K" *)
  reg signed [S_W-1:0] mem [0:DEPTH-1];
  integer n;
  initial begin
    for (n = 0; n < DEPTH; n = n + 1)
      mem[n] = {S_W{1'b0}};
  end
  always @(posedge clk) begin
    if (we)
      mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule
