// OpenLane top: rsqrt/RMSNorm unit (H=128 slice + combo Newton inv-sqrt).
`timescale 1ns / 1ps

module rmsnorm_rsqrt_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output wire done,
  output reg  alive
);
  reg signed [128*8-1:0] x_flat;
  reg signed [128*8-1:0] w_flat;
  wire signed [128*8-1:0] y_flat;

  rmsnorm128 u_rms (
    .clk(clk), .rst_n(rst_n), .en(en),
    .x_flat(x_flat), .w_flat(w_flat),
    .y_flat(y_flat), .done(done)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      x_flat <= {1023'b0, 1'b1};
      w_flat <= {1024{1'b1}};
      alive  <= 1'b0;
    end else begin
      if (en)
        x_flat <= {x_flat[1022:0], x_flat[1023] ^ x_flat[5]};
      alive <= ^y_flat ^ done;
    end
  end
endmodule
