// 16-head 0.8B mixer farm, D=4, 24 layers in time per head.
// Heads run in lockstep; they do not cut model cycles.
`timescale 1ns / 1ps

module qwen08b_heads16_d4 (
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  en,
  input  wire signed [16*32-1:0] q_flat,
  input  wire signed [16*32-1:0] k_flat,
  input  wire signed [16*32-1:0] v_flat,
  input  wire        [16*8-1:0]  g_flat,
  input  wire        [16*8-1:0]  beta_flat,
  output wire signed [16*96-1:0] o_flat,
  output wire                  done,
  output wire                  ready
);
  wire [15:0] h_done;
  wire [15:0] h_ready;
  reg  [15:0] seen;
  reg         done_r;

  genvar h;
  generate
    for (h = 0; h < 16; h = h + 1) begin : heads
      qwen08b_delta_farm #(.D(4), .N_LAYERS(24)) u_farm (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .q_flat(q_flat[h*32 +: 32]),
        .k_flat(k_flat[h*32 +: 32]),
        .v_flat(v_flat[h*32 +: 32]),
        .g(g_flat[h*8 +: 8]),
        .beta(beta_flat[h*8 +: 8]),
        .o_flat(o_flat[h*96 +: 96]),
        .done(h_done[h]),
        .ready(h_ready[h])
      );
    end
  endgenerate

  assign done  = done_r;
  assign ready = &h_ready;

  always @(posedge clk) begin
    done_r <= 1'b0;
    if (!rst_n)
      seen <= 16'd0;
    else if (en)
      seen <= 16'd0;
    else begin
      seen <= seen | h_done;
      if (&(seen | h_done) && seen != 16'hFFFF)
        done_r <= 1'b1;
    end
  end
endmodule
