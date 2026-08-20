// STA / OpenLane stand-in for rtl/delta_s_col_ram.v.
// Same 1-cycle registered 1R1W timing. Depth is two words — the compiler
// macro holds D×D state; this file exists so OpenLane closes the PE, not
// 128×128 bits of inferred SRAM as flops.
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
  reg signed [S_W-1:0] mem [0:1];
  always @(posedge clk) begin
    if (we)
      mem[waddr[0]] <= wdata;
    rdata <= mem[raddr[0]];
  end
endmodule
