// Via-tap mat-vec: x broadcast, each weight bit is a digital via (AND/shift).
// Same 8x8 4-bit W as ffn_tile_8x8_b4_reg. y = W_int @ x (integer).
// Generated; do not edit. pe_xbar/emit.py
`timescale 1ns / 1ps

// N-input signed adder. The loop is a reduction; synth maps it to a tree.
module add_tree #(
  parameter integer N = 128,
  parameter integer W = 20
) (
  input  wire signed [N*W-1:0] xs,
  output reg  signed [W-1:0]   y
);
  integer k;
  reg signed [W-1:0] acc;
  always @* begin
    acc = {W{1'b0}};
    for (k = 0; k < N; k = k + 1)
      acc = acc + $signed(xs[k*W +: W]);
    y = acc;
  end
endmodule

module ffn_rom_tap #(
  parameter integer ROWS = 8,
  parameter integer COLS = 8,
  parameter integer IN_W = 8,
  parameter integer W_W  = 4,
  parameter integer ACC_W = 15
) (
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output wire signed [ROWS*ACC_W-1:0] y_flat
);
  localparam [31:0] W_ROW0 = 32'h11f911d0;
  localparam [31:0] W_ROW1 = 32'h009f51e1;
  localparam [31:0] W_ROW2 = 32'ha2b5924f;
  localparam [31:0] W_ROW3 = 32'h19be9242;
  localparam [31:0] W_ROW4 = 32'hc194d21d;
  localparam [31:0] W_ROW5 = 32'h63f27504;
  localparam [31:0] W_ROW6 = 32'hdf1ad127;
  localparam [31:0] W_ROW7 = 32'h720fa3b3;
  localparam [255:0] WROM = {
    W_ROW7,
    W_ROW6,
    W_ROW5,
    W_ROW4,
    W_ROW3,
    W_ROW2,
    W_ROW1,
    W_ROW0
  };

// Binary two's-complement taps: each weight bit is a via that injects ±(x<<b).
function automatic signed [IN_W+4-1:0] tap_mul4;
  input signed [IN_W-1:0] x;
  input signed [4-1:0] w;
  reg signed [IN_W+4-1:0] xs;
  reg signed [IN_W+4-1:0] acc;
  begin
    xs = {{ 4{x[IN_W-1]} }, x};
    acc = 0;
    if (w[0]) acc = acc + (xs <<< 0);
    if (w[1]) acc = acc + (xs <<< 1);
    if (w[2]) acc = acc + (xs <<< 2);
    if (w[3]) acc = acc - (xs <<< 3);
    tap_mul4 = acc;
  end
endfunction

  wire signed [IN_W-1:0] x [0:COLS-1];
  genvar gi, gj;
  generate
    for (gi = 0; gi < COLS; gi = gi + 1) begin : g_unpack_x
      assign x[gi] = $signed(x_flat[gi*IN_W +: IN_W]);
    end
  endgenerate

  generate
    for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
      wire signed [ACC_W-1:0] prods [0:COLS-1];
      wire signed [COLS*ACC_W-1:0] prods_flat;
      wire signed [ACC_W-1:0] acc;
      for (gj = 0; gj < COLS; gj = gj + 1) begin : g_col
        wire signed [IN_W+W_W-1:0] p_raw;
        assign p_raw = tap_mul4(x[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
        assign prods[gj] = {{ (ACC_W-(IN_W+W_W)){p_raw[IN_W+W_W-1]} }, p_raw};
        assign prods_flat[gj*ACC_W +: ACC_W] = prods[gj];
      end
      add_tree #(.N(COLS), .W(ACC_W)) u_tree (
        .xs(prods_flat),
        .y(acc)
      );
      assign y_flat[gi*ACC_W +: ACC_W] = acc;
    end
  endgenerate

endmodule

// Registered wrapper for STA / PnR.
module ffn_rom_tap_reg (
  input  wire clk,
  input  wire signed [64-1:0] x_flat,
  output reg  signed [120-1:0] y_flat
);
  reg  signed [64-1:0] x_q;
  wire signed [120-1:0] y_c;
  ffn_rom_tap u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
