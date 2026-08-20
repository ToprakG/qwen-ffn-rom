// OpenLane top: one 8×8 FFN via-tap plus a 128-input farm adder-tree slice.
`timescale 1ns / 1ps

module ffn_tap_adder_sta (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  output reg  done,
  output reg  alive
);
  reg signed [63:0] x_flat;
  wire signed [119:0] tap_y;
  ffn_rom_tap u_tap (
    .x_flat(x_flat),
    .y_flat(tap_y)
  );

  reg signed [128*20-1:0] farm_xs;
  wire signed [19:0] farm_y;
  add_tree_bal #(.N(128), .W(20)) u_farm (
    .xs(farm_xs),
    .y(farm_y)
  );

  localparam [0:0] ST_IDLE = 1'd0;
  localparam [0:0] ST_GO   = 1'd1;
  reg st;
  integer k;

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st      <= ST_IDLE;
      x_flat  <= 64'sd1;
      farm_xs <= {2560{1'b0}};
      alive   <= 1'b0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            x_flat <= {x_flat[62:0], x_flat[63] ^ x_flat[5]};
            st     <= ST_GO;
          end
        end
        ST_GO: begin
          for (k = 0; k < 8; k = k + 1)
            farm_xs[k*20 +: 20] <= {{5{tap_y[k*15+14]}}, tap_y[k*15 +: 15]};
          for (k = 8; k < 128; k = k + 1)
            farm_xs[k*20 +: 20] <= farm_xs[(k-1)*20 +: 20] ^ farm_y;
          alive <= ^tap_y ^ ^farm_y;
          done  <= 1'b1;
          st    <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
