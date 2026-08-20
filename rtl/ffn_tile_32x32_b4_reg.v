// Hardwired mat-vec tile 32x32, signed 4-bit weights.
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
  parameter integer ROWS = 32,
  parameter integer COLS = 32,
  parameter integer IN_W = 8,
  parameter integer W_W  = 4,
  parameter integer ACC_W = 17
) (
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output wire signed [ROWS*ACC_W-1:0] y_flat
);
  localparam [127:0] W_ROW0 = 128'h2d102002511f000c3111001011f911d0;
  localparam [127:0] W_ROW1 = 128'h00f007effd100d0e1001010100d021f0;
  localparam [127:0] W_ROW2 = 128'hd2551bb1b03712ee1ff64d5bb1c4a13f;
  localparam [127:0] W_ROW3 = 128'h3ef2bdf51792a005f45403151abea232;
  localparam [127:0] W_ROW4 = 128'hfe1f45d3a7213fd3dfd011dfd1b3d21e;
  localparam [127:0] W_ROW5 = 128'h1d011f3033e0dff270d0a4e132f14302;
  localparam [127:0] W_ROW6 = 128'hfd10cc12fc0e7b0f062eec22df1bd116;
  localparam [127:0] W_ROW7 = 128'he07e19c021d30cff2fe022c0310fd1e2;
  localparam [127:0] W_ROW8 = 128'h000000000040f010010fd4e00410d009;
  localparam [127:0] W_ROW9 = 128'h10ff00300ff09cef0f133c00f100f011;
  localparam [127:0] W_ROW10 = 128'hffdfe1f0c9bc61b20e1e00fd9d021dec;
  localparam [127:0] W_ROW11 = 128'h10001922fe000f1f2ff2203100001f00;
  localparam [127:0] W_ROW12 = 128'hf1ef0bf2f322acffece36f112cf0c107;
  localparam [127:0] W_ROW13 = 128'h03f127ffce0f20200f02110ef1011dee;
  localparam [127:0] W_ROW14 = 128'h00f0000010e0f0100f0010f01f009003;
  localparam [127:0] W_ROW15 = 128'hd503e45fc9021161cd25bf3fec215aef;
  localparam [127:0] W_ROW16 = 128'h41f1d3fd20fe0e302121122f742e131f;
  localparam [127:0] W_ROW17 = 128'h3e4211e12e373c5103d23f5f12ef0de0;
  localparam [127:0] W_ROW18 = 128'hfed3d33b7f5fbcdd0f4fe6b4e04f3520;
  localparam [127:0] W_ROW19 = 128'h00e1fef029200e302f111fff320021ff;
  localparam [127:0] W_ROW20 = 128'he2e300fc010cd02d0ef00ee2e5ff7ff1;
  localparam [127:0] W_ROW21 = 128'h0f30b57fc3012344d902221fd312dcfe;
  localparam [127:0] W_ROW22 = 128'hda6d90e0fbffd0de046ee0ee205ede0b;
  localparam [127:0] W_ROW23 = 128'hc43e026eabe70b11c4e11011ecf4bc11;
  localparam [127:0] W_ROW24 = 128'h00001ef011100f0f11020f3090fff1fd;
  localparam [127:0] W_ROW25 = 128'hce61090ffaeeedfe21e44d1ee3ff2010;
  localparam [127:0] W_ROW26 = 128'hf09f240e5fd6efd0edfe3002d0df2ccf;
  localparam [127:0] W_ROW27 = 128'h1f420f1ff1f1f721f20f1f421f119f1d;
  localparam [127:0] W_ROW28 = 128'h3c6012ae1bf3df001d0ce5a1ec91fef4;
  localparam [127:0] W_ROW29 = 128'hd2b0b0ffb12493e0fe42fc1edd000136;
  localparam [127:0] W_ROW30 = 128'h13eff1ee00fc101e11231fe200701101;
  localparam [127:0] W_ROW31 = 128'h102111501900f21010f0e12120e1021f;
  localparam [4095:0] WROM = {
    W_ROW31,
    W_ROW30,
    W_ROW29,
    W_ROW28,
    W_ROW27,
    W_ROW26,
    W_ROW25,
    W_ROW24,
    W_ROW23,
    W_ROW22,
    W_ROW21,
    W_ROW20,
    W_ROW19,
    W_ROW18,
    W_ROW17,
    W_ROW16,
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
  input  wire signed [256-1:0] x_flat,
  output reg  signed [544-1:0] y_flat
);
  reg  signed [256-1:0] x_q;
  wire signed [544-1:0] y_c;
  ffn_tile u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
