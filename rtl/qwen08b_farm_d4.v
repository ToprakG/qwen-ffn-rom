`timescale 1ns / 1ps

module qwen08b_farm_d4 (
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 en,
  input  wire signed [4*8-1:0] q_flat,
  input  wire signed [4*8-1:0] k_flat,
  input  wire signed [4*8-1:0] v_flat,
  input  wire        [7:0]    g,
  input  wire        [7:0]    beta,
  output wire signed [4*24-1:0] o_flat,
  output wire                 done,
  output wire                 ready
);
  qwen08b_delta_farm #(.D(4), .N_LAYERS(24)) u_farm (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .q_flat(q_flat),
    .k_flat(k_flat),
    .v_flat(v_flat),
    .g(g),
    .beta(beta),
    .o_flat(o_flat),
    .done(done),
    .ready(ready)
  );
endmodule
