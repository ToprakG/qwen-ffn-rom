`timescale 1ns / 1ps

module gated_delta_d128_bram (
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    en,
  input  wire signed [128*8-1:0] q_flat,
  input  wire signed [128*8-1:0] k_flat,
  input  wire signed [128*8-1:0] v_flat,
  input  wire        [7:0]       g,
  input  wire        [7:0]       beta,
  output wire signed [128*24-1:0] o_flat,
  output wire                    done,
  output wire                    ready
);
  gated_delta_bram #(.D(128), .N_LAYERS(1)) u_pe (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .layer(1'b0),
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
