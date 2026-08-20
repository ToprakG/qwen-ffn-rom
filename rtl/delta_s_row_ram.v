// Row-major S RAM. One address = one row of D packed S_W values.
// Simple dual-port (independent read/write) so Yosys/Vivado infer block RAM.
// Do not loop-clear this memory in an always block — that becomes flops.
`timescale 1ns / 1ps

module delta_s_row_ram #(
  parameter integer DEPTH = 16,
  parameter integer WIDTH = 256
) (
  input  wire                         clk,
  input  wire                         we,
  input  wire [$clog2(DEPTH)-1:0]     waddr,
  input  wire [WIDTH-1:0]             wdata,
  input  wire [$clog2(DEPTH)-1:0]     raddr,
  output reg  [WIDTH-1:0]             rdata
);
  (* ram_style = "block" *)
  reg [WIDTH-1:0] mem [0:DEPTH-1];

  integer n;
  initial begin
    for (n = 0; n < DEPTH; n = n + 1)
      mem[n] = {WIDTH{1'b0}};
  end

  always @(posedge clk) begin
    if (we)
      mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule
