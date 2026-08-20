// Handshake around the combo 8x8 via-tap: 1 compute cycle after en.
// Used to count FFN clocks (the combo tile itself is 0 sequential cycles).
`timescale 1ns / 1ps

module ffn_tap_unit (
  input  wire                clk,
  input  wire                rst_n,
  input  wire                en,
  input  wire signed [63:0]  x_flat,
  output reg  signed [119:0] y_flat,
  output reg                 done
);
  reg signed [63:0] x_r;
  wire signed [119:0] y_c;
  ffn_rom_tap u_tap (
    .x_flat(x_r),
    .y_flat(y_c)
  );

  localparam [0:0] ST_IDLE = 1'd0;
  localparam [0:0] ST_GO   = 1'd1;
  reg st;

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st     <= ST_IDLE;
      x_r    <= 64'sd0;
      y_flat <= 120'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            x_r <= x_flat;
            st  <= ST_GO;
          end
        end
        ST_GO: begin
          y_flat <= y_c;
          done   <= 1'b1;
          st     <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
