// SwiGLU output stage folded into the 8x8 tap handshake.
// Cycle count stays 2 (same as ffn_tap_unit): capture, then combo tap+SiLU+gate.
// y = sat8( (silu_q3(sat8(W@x >> 7)) * up) >> 7 )
// Bit-exact vs quant/silu_int.py swiglu_out.
`timescale 1ns / 1ps

module ffn_tap_swiglu (
  input  wire                clk,
  input  wire                rst_n,
  input  wire                en,
  input  wire signed [63:0]  x_flat,
  input  wire signed [63:0]  up_flat,
  output reg  signed [63:0]  y_flat,
  output reg                 done
);
  localparam [0:0] ST_IDLE = 1'd0;
  localparam [0:0] ST_GO   = 1'd1;

  reg signed [63:0] x_r;
  reg signed [63:0] up_r;
  wire signed [119:0] gate_acc;
  ffn_rom_tap u_tap (
    .x_flat(x_r),
    .y_flat(gate_acc)
  );

  wire signed [7:0] up_i [0:7];
  wire signed [14:0] acc [0:7];
  wire signed [7:0]  g8  [0:7];
  wire signed [7:0]  s8  [0:7];
  wire signed [7:0]  y8  [0:7];
  genvar gi;
  generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : lanes
      assign up_i[gi] = up_r[gi*8 +: 8];
      assign acc[gi]  = gate_acc[gi*15 +: 15];
      wire signed [31:0] gsh = acc[gi] >>> 7;
      assign g8[gi] = (gsh > 32'sd127) ? 8'sd127 :
                      (gsh < -32'sd128) ? -8'sd128 : gsh[7:0];
      silu_lut u_s (.x(g8[gi]), .y(s8[gi]));
      wire signed [31:0] p = s8[gi] * up_i[gi];
      wire signed [31:0] psh = p >>> 7;
      assign y8[gi] = (psh > 32'sd127) ? 8'sd127 :
                      (psh < -32'sd128) ? -8'sd128 : psh[7:0];
    end
  endgenerate

  reg st;

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st     <= ST_IDLE;
      x_r    <= 64'sd0;
      up_r   <= 64'sd0;
      y_flat <= 64'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            x_r  <= x_flat;
            up_r <= up_flat;
            st   <= ST_GO;
          end
        end
        ST_GO: begin
          y_flat <= {y8[7], y8[6], y8[5], y8[4], y8[3], y8[2], y8[1], y8[0]};
          done   <= 1'b1;
          st     <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
