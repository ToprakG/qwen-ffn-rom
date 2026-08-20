// Balanced signed adder tree. Heap layout, pad to power-of-two.
// Sequential acc loops become ripple at N=128; this is log2(N)  W-bit adds.
`timescale 1ns / 1ps

module add_tree_bal #(
  parameter integer N = 16,
  parameter integer W = 32
) (
  input  wire signed [N*W-1:0] xs,
  output wire signed [W-1:0]   y
);
  localparam integer NP = 2 ** $clog2(N);
  wire signed [W-1:0] tree [0:2*NP-2];
  genvar gi;
  generate
    for (gi = 0; gi < NP; gi = gi + 1) begin : leaves
      if (gi < N)
        assign tree[NP-1+gi] = xs[gi*W +: W];
      else
        assign tree[NP-1+gi] = {W{1'b0}};
    end
    for (gi = 0; gi < NP-1; gi = gi + 1) begin : nodes
      assign tree[gi] = tree[2*gi+1] + tree[2*gi+2];
    end
  endgenerate
  assign y = tree[0];
endmodule
