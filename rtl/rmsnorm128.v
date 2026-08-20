// RMSNorm slice at mixer width (H=128). Combo Newton rsqrt is inside.
`timescale 1ns / 1ps

module rmsnorm128 (
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    en,
  input  wire signed [128*8-1:0] x_flat,
  input  wire signed [128*8-1:0] w_flat,
  output wire signed [128*8-1:0] y_flat,
  output wire                    done
);
  rmsnorm #(.H(128)) u (
    .clk(clk), .rst_n(rst_n), .en(en),
    .x_flat(x_flat), .w_flat(w_flat),
    .y_flat(y_flat), .done(done)
  );
endmodule
