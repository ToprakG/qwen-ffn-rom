`timescale 1ns / 1ps

module attn_online_d8_p4 (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire [15:0]               seq_len,
  input  wire signed [8*8-1:0]     q_flat,
  input  wire                      wr_en,
  input  wire [15:0]               wr_t,
  input  wire [8*4-1:0]            wr_k,
  input  wire [8*4-1:0]            wr_v,
  output wire signed [8*24-1:0]    o_flat,
  output wire                      done,
  output wire                      ready
);
  attn_online #(
    .D(8), .P(4), .S_MAX(64), .N_KV(1), .N_Q_PER(1)
  ) u (
    .clk(clk), .rst_n(rst_n), .en(en), .seq_len(seq_len),
    .q_flat(q_flat), .wr_en(wr_en), .wr_t(wr_t), .wr_k(wr_k), .wr_v(wr_v),
    .o_flat(o_flat), .done(done), .ready(ready)
  );
endmodule
