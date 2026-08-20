// OpenLane top: one 27B D=128 fused mixer PE, chip pins only.
// Vectors stay on-die (LFSR stimulus) so Fmax is the combo EX, not pad delay.
`timescale 1ns / 1ps

module gated_delta_d128_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output wire done,
  output wire ready,
  output reg  alive
);
  reg signed [128*8-1:0] q_flat;
  reg signed [128*8-1:0] k_flat;
  reg signed [128*8-1:0] v_flat;
  reg        [7:0]       g;
  reg        [7:0]       beta;
  wire signed [128*24-1:0] o_flat;

  gated_delta_fused #(.D(128)) u_pe (
    .clk(clk), .rst_n(rst_n), .en(en),
    .q_flat(q_flat), .k_flat(k_flat), .v_flat(v_flat),
    .g(g), .beta(beta),
    .o_flat(o_flat), .done(done), .ready(ready)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      q_flat <= {1023'b0, 1'b1};
      k_flat <= {1022'b0, 2'b11};
      v_flat <= {1021'b0, 3'b101};
      g      <= 8'd17;
      beta   <= 8'd9;
      alive  <= 1'b0;
    end else begin
      if (en || !ready) begin
        q_flat <= {q_flat[1022:0], q_flat[1023] ^ q_flat[21] ^ q_flat[1]};
        k_flat <= {k_flat[1022:0], k_flat[1023] ^ k_flat[15]};
        v_flat <= {v_flat[1022:0], v_flat[1023] ^ v_flat[8]};
        g      <= g + 8'd1;
        beta   <= beta + 8'd3;
      end
      alive <= ^o_flat ^ done ^ ready;
    end
  end
endmodule
