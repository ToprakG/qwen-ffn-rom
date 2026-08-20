// Integer RMSNorm, width H: y_i = sat8((x_i * w_i * inv) >>> 16)
// inv from combo Newton rsqrt. 2-cycle handshake (capture, mul+done).
// Bit-exact vs quant/rsqrt_int.py rmsnorm_nr.
`timescale 1ns / 1ps

module rmsnorm #(
  parameter integer H = 16
) (
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    en,
  input  wire signed [H*8-1:0]   x_flat,
  input  wire signed [H*8-1:0]   w_flat,
  output reg  signed [H*8-1:0]   y_flat,
  output reg                     done
);
  localparam [0:0] ST_IDLE = 1'd0;
  localparam [0:0] ST_MUL  = 1'd1;

  wire signed [7:0] x_i [0:H-1];
  wire signed [7:0] w_i [0:H-1];
  genvar gi;
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : unpack
      assign x_i[gi] = x_flat[gi*8 +: 8];
      assign w_i[gi] = w_flat[gi*8 +: 8];
    end
  endgenerate

  integer k;
  reg  signed [7:0]  x_r  [0:H-1];
  reg  signed [7:0]  w_r  [0:H-1];
  wire signed [H*32-1:0] sq_flat;
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : gsqr
      assign sq_flat[gi*32 +: 32] = x_r[gi] * x_r[gi];
    end
  endgenerate
  wire [31:0] acc_r;
  add_tree_bal #(.N(H), .W(32)) u_sumsq (.xs(sq_flat), .y(acc_r));

  wire [16:0] inv;
  inv_rsqrt u_inv (.x(acc_r), .inv(inv));

  wire signed [7:0] ysat [0:H-1];
  wire signed [31:0] inv_s = $signed({15'd0, inv});
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : lanes
      wire signed [47:0] prod48 = $signed(x_r[gi]) * $signed(w_r[gi]) * inv_s;
      wire signed [31:0] sh = prod48 >>> 16;
      assign ysat[gi] = (sh > 32'sd127) ? 8'sd127 :
                        (sh < -32'sd128) ? -8'sd128 : sh[7:0];
    end
  endgenerate

  reg st;

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st     <= ST_IDLE;
      y_flat <= {H*8{1'b0}};
      for (k = 0; k < H; k = k + 1) begin
        x_r[k] <= 8'sd0;
        w_r[k] <= 8'sd0;
      end
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (k = 0; k < H; k = k + 1) begin
              x_r[k] <= x_i[k];
              w_r[k] <= w_i[k];
            end
            st <= ST_MUL;
          end
        end
        ST_MUL: begin
          for (k = 0; k < H; k = k + 1)
            y_flat[k*8 +: 8] <= ysat[k];
          done <= 1'b1;
          st   <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
