// D=16 instance of the D-parallel Gated DeltaNet PE (16 MACs).
`timescale 1ns / 1ps

module gated_delta_d16_par (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire signed [16*8-1:0]    q_flat,
  input  wire signed [16*8-1:0]    k_flat,
  input  wire signed [16*8-1:0]    v_flat,
  input  wire        [7:0]         g,
  input  wire        [7:0]         beta,
  output wire signed [16*24-1:0]   o_flat,
  output wire                      done
);
  gated_delta_dpar #(.D(16)) u_pe (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .q_flat(q_flat),
    .k_flat(k_flat),
    .v_flat(v_flat),
    .g(g),
    .beta(beta),
    .o_flat(o_flat),
    .done(done)
  );
endmodule
