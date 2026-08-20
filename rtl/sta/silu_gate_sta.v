// OpenLane top: SiLU LUT + SwiGLU gate folded into the 8×8 tap handshake.
`timescale 1ns / 1ps

module silu_gate_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output wire done,
  output reg  alive
);
  reg signed [63:0] x_flat;
  reg signed [63:0] up_flat;
  wire signed [63:0] y_flat;

  ffn_tap_swiglu u_silu (
    .clk(clk), .rst_n(rst_n), .en(en),
    .x_flat(x_flat), .up_flat(up_flat),
    .y_flat(y_flat), .done(done)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      x_flat  <= 64'sd1;
      up_flat <= 64'sd3;
      alive   <= 1'b0;
    end else begin
      if (en) begin
        x_flat  <= {x_flat[62:0], x_flat[63] ^ x_flat[5]};
        up_flat <= {up_flat[62:0], up_flat[63] ^ up_flat[1]};
      end
      alive <= ^y_flat ^ done;
    end
  end
endmodule
