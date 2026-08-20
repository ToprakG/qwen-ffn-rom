// OpenLane top: one attention streaming-sweep unit, chip pins only.
`timescale 1ns / 1ps

module attn_sweep_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output wire done,
  output wire ready,
  output reg  alive
);
  reg signed [256*8-1:0] q_flat;
  reg [15:0] seq_len;
  wire signed [256*24-1:0] o_flat;

  attn_sweep_pe #(.D(256), .P(4), .S_MAX(16)) u_pe (
    .clk(clk), .rst_n(rst_n), .en(en),
    .seq_len(seq_len),
    .q_flat(q_flat),
    .wr_en(1'b0), .wr_t(16'd0),
    .wr_k({1024{1'b0}}), .wr_v({1024{1'b0}}),
    .o_flat(o_flat), .done(done), .ready(ready)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      q_flat  <= {2047'b0, 1'b1};
      seq_len <= 16'd8;
      alive   <= 1'b0;
    end else begin
      if (en || !ready)
        q_flat <= {q_flat[2046:0], q_flat[2047] ^ q_flat[21]};
      alive <= ^o_flat ^ done ^ ready;
    end
  end
endmodule
