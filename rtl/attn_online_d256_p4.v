// One 27B attention streaming-sweep unit: head_dim 256, 4 KV-SRAM banks.
// Chip uses P_eff=512 (256 banks × int4 pack-2); Fmax is this PE's combo
// path (D-wide int8×int4 + online softmax + last-cycle tdiv). Extra P
// copies do not deepen that path beyond a log-P reduce.
`timescale 1ns / 1ps

module attn_online_d256_p4 (
  input  wire                        clk,
  input  wire                        rst_n,
  input  wire                        en,
  input  wire [15:0]                 seq_len,
  input  wire signed [256*8-1:0]     q_flat,
  input  wire                        wr_en,
  input  wire [15:0]                 wr_t,
  input  wire [256*4-1:0]            wr_k,
  input  wire [256*4-1:0]            wr_v,
  output wire signed [256*24-1:0]    o_flat,
  output wire                        done,
  output wire                        ready
);
  attn_online #(
    .D(256), .P(4), .S_MAX(16), .N_KV(1), .N_Q_PER(1)
  ) u (
    .clk(clk), .rst_n(rst_n), .en(en), .seq_len(seq_len),
    .q_flat(q_flat), .wr_en(wr_en), .wr_t(wr_t), .wr_k(wr_k), .wr_v(wr_v),
    .o_flat(o_flat), .done(done), .ready(ready)
  );
endmodule
