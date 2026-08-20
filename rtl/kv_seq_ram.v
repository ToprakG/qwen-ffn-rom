// Packed vector SRAM bank. Same 1R1W inferred-BRAM discipline as delta_s_col_ram:
// registered read, no loop-clear in always. Address = sequence step (t / P);
// P banks in parallel give a P-lane KV sweep.
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
  (* ram_style = "block" *)
  (* ramstyle = "M20K" *)
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
