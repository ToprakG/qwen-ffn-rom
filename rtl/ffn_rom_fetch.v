// Via-ROM as memory: decode one (row,col), one MAC, scan the matrix.
// Same 8x8 4-bit W as ffn_tile_8x8_b4_reg. y = W_int @ x (integer).
// Generated; do not edit. pe_xbar/emit.py
`timescale 1ns / 1ps

module ffn_rom_fetch #(
  parameter integer ROWS = 8,
  parameter integer COLS = 8,
  parameter integer IN_W = 8,
  parameter integer W_W  = 4,
  parameter integer ACC_W = 15
) (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output reg  signed [ROWS*ACC_W-1:0] y_flat,
  output reg                       done
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

  localparam ST_IDLE = 2'd0;
  localparam ST_MAC  = 2'd1;
  localparam ST_PACK = 2'd2;

  reg  [1:0] st;
  reg  [2:0] i;
  reg  [2:0] j;
  reg signed [IN_W-1:0] x_r [0:COLS-1];
  reg signed [31:0]     acc [0:ROWS-1];

  wire [5:0] idx = {i, j};
  wire signed [W_W-1:0] w_sel = $signed(WROM[idx*W_W +: W_W]);
  wire signed [31:0]    prod  = $signed(w_sel) * $signed(x_r[j]);

  integer ci, ri;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      i <= 3'd0;
      j <= 3'd0;
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
            i <= 3'd0;
            j <= 3'd0;
            st <= ST_MAC;
          end
        end
        ST_MAC: begin
          acc[i] <= acc[i] + prod;
          if (j == 3'd7) begin
            j <= 3'd0;
            if (i == 3'd7)
              st <= ST_PACK;
            else
              i <= i + 3'd1;
          end else
            j <= j + 3'd1;
        end
        ST_PACK: begin
          for (ri = 0; ri < ROWS; ri = ri + 1)
            y_flat[ri*ACC_W +: ACC_W] <= acc[ri][ACC_W-1:0];
          done <= 1'b1;
          st <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
