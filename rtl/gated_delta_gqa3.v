// 1 K-head × 3 V-heads (Qwen GQA). Shared q, k, g, β; independent S and v.
// Three fused PEs in lockstep. Layer cycles = one PE (D+2), not 3×.
// Reusing the K-head read across 3 V-heads is why "heads in parallel"
// cuts the serial-V schedule; it does not cut below one PE's D+2.
`timescale 1ns / 1ps

module gated_delta_gqa3 #(
  parameter integer D = 16
) (
  input  wire                         clk,
  input  wire                         rst_n,
  input  wire                         en,
  input  wire signed [D*8-1:0]        q_flat,
  input  wire signed [D*8-1:0]        k_flat,
  input  wire signed [3*D*8-1:0]      v_flat,
  input  wire        [7:0]            g,
  input  wire        [7:0]            beta,
  output wire signed [3*D*24-1:0]     o_flat,
  output wire                         done,
  output wire                         ready
);
  wire done0, done1, done2;
  wire rdy0, rdy1, rdy2;

  gated_delta_fused #(.D(D)) u0 (
    .clk(clk), .rst_n(rst_n), .en(en),
    .q_flat(q_flat), .k_flat(k_flat),
    .v_flat(v_flat[0*D*8 +: D*8]),
    .g(g), .beta(beta),
    .o_flat(o_flat[0*D*24 +: D*24]),
    .done(done0), .ready(rdy0)
  );
  gated_delta_fused #(.D(D)) u1 (
    .clk(clk), .rst_n(rst_n), .en(en),
    .q_flat(q_flat), .k_flat(k_flat),
    .v_flat(v_flat[1*D*8 +: D*8]),
    .g(g), .beta(beta),
    .o_flat(o_flat[1*D*24 +: D*24]),
    .done(done1), .ready(rdy1)
  );
  gated_delta_fused #(.D(D)) u2 (
    .clk(clk), .rst_n(rst_n), .en(en),
    .q_flat(q_flat), .k_flat(k_flat),
    .v_flat(v_flat[2*D*8 +: D*8]),
    .g(g), .beta(beta),
    .o_flat(o_flat[2*D*24 +: D*24]),
    .done(done2), .ready(rdy2)
  );

  assign done  = done0;
  assign ready = rdy0 & rdy1 & rdy2;
endmodule
