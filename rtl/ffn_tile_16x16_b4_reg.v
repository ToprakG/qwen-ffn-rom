// Hardwired mat-vec tile 16x16, signed 4-bit weights.
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
  parameter integer W_W  = 4,
  parameter integer ACC_W = 16
) (
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output wire signed [ROWS*ACC_W-1:0] y_flat
);
  localparam [63:0] W_ROW0 = 64'h3111001011f911d0;
  localparam [63:0] W_ROW1 = 64'h2003f112009f51e1;
  localparam [63:0] W_ROW2 = 64'h1ff75d5ba1c4a24f;
  localparam [63:0] W_ROW3 = 64'hf565031519be9242;
  localparam [63:0] W_ROW4 = 64'hcfc011cfc194d21d;
  localparam [63:0] W_ROW5 = 64'h70d0a4e132f14302;
  localparam [63:0] W_ROW6 = 64'h073eeb22df1ad127;
  localparam [63:0] W_ROW7 = 64'h3fd03391610fb3c3;
  localparam [63:0] W_ROW8 = 64'h010fd4e00410d009;
  localparam [63:0] W_ROW9 = 64'h1e355910f1f0e012;
  localparam [63:0] W_ROW10 = 64'h0e1e00fd9d021dec;
  localparam [63:0] W_ROW11 = 64'h4ee45f7400f02df1;
  localparam [63:0] W_ROW12 = 64'hece36f112cf0c107;
  localparam [63:0] W_ROW13 = 64'h0df413fcf30139cb;
  localparam [63:0] W_ROW14 = 64'h0f0010f01f009003;
  localparam [63:0] W_ROW15 = 64'hbd25bf3fec3159ef;
  localparam [1023:0] WROM = {
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
// Signed 4-bit symmetric range -7..7.
// Each unique weight maps to one expression (inlined at each call site).
function automatic signed [IN_W+4-1:0] csd_mul4;
  input signed [IN_W-1:0] x;
  input signed [4-1:0] w;
  reg signed [IN_W+4-1:0] xs;
  begin
    xs = {{ 4{x[IN_W-1]} }, x};
    case (w)
      4'h9: csd_mul4 = xs - (xs <<< 3);  // -7
      4'ha: csd_mul4 = (xs <<< 1) - (xs <<< 3);  // -6
      4'hb: csd_mul4 = -xs - (xs <<< 2);  // -5
      4'hc: csd_mul4 = -(xs <<< 2);  // -4
      4'hd: csd_mul4 = xs - (xs <<< 2);  // -3
      4'he: csd_mul4 = -(xs <<< 1);  // -2
      4'hf: csd_mul4 = -xs;  // -1
      4'h0: csd_mul4 = 0;  // 0
      4'h1: csd_mul4 = xs;  // 1
      4'h2: csd_mul4 = (xs <<< 1);  // 2
      4'h3: csd_mul4 = -xs + (xs <<< 2);  // 3
      4'h4: csd_mul4 = (xs <<< 2);  // 4
      4'h5: csd_mul4 = xs + (xs <<< 2);  // 5
      4'h6: csd_mul4 = -(xs <<< 1) + (xs <<< 3);  // 6
      4'h7: csd_mul4 = -xs + (xs <<< 3);  // 7
      default: csd_mul4 = 0;
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
        assign p_raw = csd_mul4(x[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
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
  output reg  signed [256-1:0] y_flat
);
  reg  signed [128-1:0] x_q;
  wire signed [256-1:0] y_c;
  ffn_tile u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
