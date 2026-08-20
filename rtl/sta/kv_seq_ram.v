// STA / OpenLane stand-in for rtl/kv_seq_ram.v.
// Same 1-cycle registered 1R1W. Two-word body — KV-SRAM is a compiler macro.
`timescale 1ns / 1ps

module kv_seq_ram #(
  parameter integer DEPTH  = 16,
  parameter integer WIDTH  = 32,
  parameter integer ADDR_W = $clog2(DEPTH)
) (
  input  wire                 clk,
  input  wire                 we,
  input  wire [ADDR_W-1:0]    waddr,
  input  wire [WIDTH-1:0]     wdata,
  input  wire [ADDR_W-1:0]    raddr,
  output reg  [WIDTH-1:0]     rdata
);
  reg [WIDTH-1:0] mem [0:1];
  always @(posedge clk) begin
    if (we)
      mem[waddr[0]] <= wdata;
    rdata <= mem[raddr[0]];
  end
endmodule
