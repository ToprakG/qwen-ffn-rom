// Hardwired mat-vec tile 16x16, signed 3-bit weights.
// y = W_int * x  (integer; per-row scale is outside this module).
// Each weight is a CSD shift-add (csd_mul*), not a general multiplier.
// Generated from Qwen3.5-0.8B layer0 down_proj (per-output-row quant).
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

module ffn_tile #(
  parameter integer ROWS = 16,
  parameter integer COLS = 16,
  parameter integer IN_W = 8,
  parameter integer W_W  = 3,
  parameter integer ACC_W = 15
) (
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output wire signed [ROWS*ACC_W-1:0] y_flat
);
  localparam [47:0] W_ROW0 = 48'h20000003d078;
  localparam [47:0] W_ROW1 = 48'h201041028478;
  localparam [47:0] W_ROW2 = 48'h23b5d6a72a50;
  localparam [47:0] W_ROW3 = 48'h09a042377a51;
  localparam [47:0] W_ROW4 = 48'hdf0230c6ae4f;
  localparam [47:0] W_ROW5 = 48'h638cb9240441;
  localparam [47:0] W_ROW6 = 48'h0cff89fc6e0b;
  localparam [47:0] W_ROW7 = 48'h3f8268647c71;
  localparam [47:0] W_ROW8 = 48'h040eb8080e05;
  localparam [47:0] W_ROW9 = 48'h1ca540e40e01;
  localparam [47:0] W_ROW10 = 48'h1c7007bc11fe;
  localparam [47:0] W_ROW11 = 48'h5fa5da0003c0;
  localparam [47:0] W_ROW12 = 48'hfb9600380c03;
  localparam [47:0] W_ROW13 = 48'h1c2246e41376;
  localparam [47:0] W_ROW14 = 48'h000038000a01;
  localparam [47:0] W_ROW15 = 48'hdcadc8f89578;
  localparam [767:0] WROM = {
    W_ROW15,
    W_ROW14,
    W_ROW13,
    W_ROW12,
    W_ROW11,
    W_ROW10,
    W_ROW9,
    W_ROW8,
    W_ROW7,
    W_ROW6,
    W_ROW5,
    W_ROW4,
    W_ROW3,
    W_ROW2,
    W_ROW1,
    W_ROW0
  };

// Constant-coefficient multiply: CSD recoding → shift-add/sub, never a general *.
// Signed 3-bit symmetric range -3..3.
// Each unique weight maps to one expression (inlined at each call site).
function automatic signed [IN_W+3-1:0] csd_mul3;
  input signed [IN_W-1:0] x;
  input signed [3-1:0] w;
  reg signed [IN_W+3-1:0] xs;
  begin
    xs = {{ 3{x[IN_W-1]} }, x};
    case (w)
      3'h5: csd_mul3 = xs - (xs <<< 2);  // -3
      3'h6: csd_mul3 = -(xs <<< 1);  // -2
      3'h7: csd_mul3 = -xs;  // -1
      3'h0: csd_mul3 = 0;  // 0
      3'h1: csd_mul3 = xs;  // 1
      3'h2: csd_mul3 = (xs <<< 1);  // 2
      3'h3: csd_mul3 = -xs + (xs <<< 2);  // 3
      default: csd_mul3 = 0;
    endcase
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
        assign p_raw = csd_mul3(x[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
        assign prods[gj] = {{(ACC_W-(IN_W+W_W)){p_raw[IN_W+W_W-1]}}, p_raw};
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
// Registered wrapper for STA / PnR. Combo core is `ffn_tile`.
module ffn_tile_reg (
  input  wire clk,
  input  wire signed [128-1:0] x_flat,
  output reg  signed [240-1:0] y_flat
);
  reg  signed [128-1:0] x_q;
  wire signed [240-1:0] y_c;
  ffn_tile u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
