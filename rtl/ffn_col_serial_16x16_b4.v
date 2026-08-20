// Bit-serial CSD column: 8 phases of x bits, CSD taps, digital partials.
// 16x16 4-bit W. y = W_int @ x (integer).
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

module ffn_col_serial #(
  parameter integer ROWS = 16,
  parameter integer COLS = 16,
  parameter integer IN_W = 8,
  parameter integer W_W  = 4,
  parameter integer ACC_W = 16
) (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output reg  signed [ROWS*ACC_W-1:0] y_flat,
  output reg                       done
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

  localparam ST_IDLE = 1'b0;
  localparam ST_RUN  = 1'b1;

  reg        st;
  reg  [2:0] k;
  reg signed [IN_W-1:0]  x_r  [0:COLS-1];
  reg signed [31:0]      acc  [0:ROWS-1];

  wire signed [IN_W-1:0] xb [0:COLS-1];
  genvar gx;
  generate
    for (gx = 0; gx < COLS; gx = gx + 1) begin : g_xb
      assign xb[gx] = {{ (IN_W-1){1'b0} }, x_r[gx][k]};
    end
  endgenerate

  wire signed [ACC_W-1:0] part [0:ROWS-1];
  wire signed [31:0]      contrib [0:ROWS-1];
  wire signed [ROWS*ACC_W-1:0] y_next_flat;
  genvar gi, gj;
  generate
    for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
      wire signed [ACC_W-1:0] prods [0:COLS-1];
      wire signed [COLS*ACC_W-1:0] prods_flat;
      for (gj = 0; gj < COLS; gj = gj + 1) begin : g_col
        wire signed [IN_W+W_W-1:0] p_raw;
        assign p_raw = csd_mul4(xb[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
        assign prods[gj] = {{ (ACC_W-(IN_W+W_W)){p_raw[IN_W+W_W-1]} }, p_raw};
        assign prods_flat[gj*ACC_W +: ACC_W] = prods[gj];
      end
      add_tree #(.N(COLS), .W(ACC_W)) u_tree (
        .xs(prods_flat),
        .y(part[gi])
      );
      wire signed [31:0] part_ext = {{ (32-ACC_W){part[gi][ACC_W-1]} }, part[gi]};
      assign contrib[gi] = (k == 3'd7) ? -(part_ext <<< k) : (part_ext <<< k);
      wire signed [ACC_W-1:0] y_next = acc[gi] + contrib[gi];
      assign y_next_flat[gi*ACC_W +: ACC_W] = y_next;
    end
  endgenerate

  integer ci, ri;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      k <= 3'd0;
      y_flat <= {ROWS * ACC_W{1'b0}};
      for (ci = 0; ci < COLS; ci = ci + 1)
        x_r[ci] <= {IN_W{1'b0}};
      for (ri = 0; ri < ROWS; ri = ri + 1)
        acc[ri] <= 32'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (ci = 0; ci < COLS; ci = ci + 1)
              x_r[ci] <= $signed(x_flat[ci*IN_W +: IN_W]);
            for (ri = 0; ri < ROWS; ri = ri + 1)
              acc[ri] <= 32'sd0;
            k <= 3'd0;
            st <= ST_RUN;
          end
        end
        ST_RUN: begin
          for (ri = 0; ri < ROWS; ri = ri + 1)
            acc[ri] <= acc[ri] + contrib[ri];
          if (k == 3'd7) begin
            y_flat <= y_next_flat;
            done <= 1'b1;
            st <= ST_IDLE;
          end else
            k <= k + 3'd1;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
