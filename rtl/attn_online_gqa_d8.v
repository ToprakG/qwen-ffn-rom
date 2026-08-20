`timescale 1ns / 1ps

// 4 KV heads × 6 Q heads. Cycles must equal the 1-head PE (K reuse).
module attn_online_gqa_d8 (
  input  wire                       clk,
  input  wire                       rst_n,
  input  wire                       en,
  input  wire [15:0]                seq_len,
  input  wire signed [24*8*8-1:0]   q_flat,
  input  wire                       wr_en,
  input  wire [15:0]                wr_t,
  input  wire [4*8*4-1:0]           wr_k,
  input  wire [4*8*4-1:0]           wr_v,
  output wire signed [24*8*24-1:0]  o_flat,
  output wire                       done,
  output wire                       ready
);
  attn_online #(
    .D(8), .P(4), .S_MAX(32), .N_KV(4), .N_Q_PER(6)
  ) u (
    .clk(clk), .rst_n(rst_n), .en(en), .seq_len(seq_len),
    .q_flat(q_flat), .wr_en(wr_en), .wr_t(wr_t), .wr_k(wr_k), .wr_v(wr_v),
    .o_flat(o_flat), .done(done), .ready(ready)
  );
endmodule
