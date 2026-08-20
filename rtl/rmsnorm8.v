// H=8 wrapper around parameterized rmsnorm. Same ports as the old unit.
`timescale 1ns / 1ps

module rmsnorm8 (
  input  wire                clk,
  input  wire                rst_n,
  input  wire                en,
  input  wire signed [63:0]  x_flat,
  input  wire signed [63:0]  w_flat,
  output wire signed [63:0]  y_flat,
  output wire                done
);
  rmsnorm #(.H(8)) u (
    .clk(clk), .rst_n(rst_n), .en(en),
    .x_flat(x_flat), .w_flat(w_flat), .y_flat(y_flat), .done(done)
  );
endmodule
