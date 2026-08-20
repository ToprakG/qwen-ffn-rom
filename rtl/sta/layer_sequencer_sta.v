// OpenLane top: sequencer/handshake only. PE dones are 1-cycle stubs.
`timescale 1ns / 1ps

module layer_sequencer_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output wire done,
  output wire ready,
  output reg  alive
);
  wire rms_en, body_en, ffn_en;
  reg  rms_done, body_done, ffn_done, body_ready;
  reg  mix_sel;

  layer_sequencer u_seq (
    .clk(clk), .rst_n(rst_n), .en(en),
    .mix_sel(mix_sel),
    .rms_done(rms_done), .body_done(body_done),
    .body_ready(body_ready), .ffn_done(ffn_done),
    .rms_en(rms_en), .body_en(body_en), .ffn_en(ffn_en),
    .done(done), .ready(ready)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      rms_done    <= 1'b0;
      body_done   <= 1'b0;
      ffn_done    <= 1'b0;
      body_ready  <= 1'b1;
      mix_sel     <= 1'b1;
      alive       <= 1'b0;
    end else begin
      rms_done   <= rms_en;
      body_done  <= body_en;
      ffn_done   <= ffn_en;
      mix_sel    <= mix_sel ^ en;
      alive      <= done ^ ready ^ rms_en ^ body_en ^ ffn_en;
    end
  end
endmodule
