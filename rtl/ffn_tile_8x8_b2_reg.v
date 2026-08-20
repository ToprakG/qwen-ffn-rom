// Hardwired mat-vec tile 8x8, signed 2-bit weights.
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
  parameter integer ROWS = 8,
  parameter integer COLS = 8,
  parameter integer IN_W = 8,
  parameter integer W_W  = 2,
  parameter integer ACC_W = 13
) (
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output wire signed [ROWS*ACC_W-1:0] y_flat
);
  localparam [15:0] W_ROW0 = 16'h0300;
  localparam [15:0] W_ROW1 = 16'h0c40;
  localparam [15:0] W_ROW2 = 16'hcdc4;
  localparam [15:0] W_ROW3 = 16'h3cc4;
  localparam [15:0] W_ROW4 = 16'hcd00;
  localparam [15:0] W_ROW5 = 16'h4051;
  localparam [15:0] W_ROW6 = 16'h0301;
  localparam [15:0] W_ROW7 = 16'h40cc;
  localparam [127:0] WROM = {
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
// Signed 2-bit symmetric range -1..1.
// Each unique weight maps to one expression (inlined at each call site).
function automatic signed [IN_W+2-1:0] csd_mul2;
  input signed [IN_W-1:0] x;
  input signed [2-1:0] w;
  reg signed [IN_W+2-1:0] xs;
  begin
    xs = {{ 2{x[IN_W-1]} }, x};
    case (w)
      2'h3: csd_mul2 = -xs;  // -1
      2'h0: csd_mul2 = 0;  // 0
      2'h1: csd_mul2 = xs;  // 1
      default: csd_mul2 = 0;
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
        assign p_raw = csd_mul2(x[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
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
  input  wire signed [64-1:0] x_flat,
  output reg  signed [104-1:0] y_flat
);
  reg  signed [64-1:0] x_q;
  wire signed [104-1:0] y_c;
  ffn_tile u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
