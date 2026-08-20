// 16-cycle restoring integer square root. y = floor(sqrt(x)), x unsigned 32-bit.
`timescale 1ns / 1ps

module isqrt32 (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        en,
  input  wire [31:0] x,
  output reg  [15:0] y,
  output reg         done
);
  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_RUN  = 2'd1;

  reg [1:0]  st;
  reg [3:0]  i;
  reg [31:0] x_r;
  reg [31:0] rem;
  reg [15:0] root;
  wire [31:0] rem_n = {rem[29:0], x_r[31:30]};
  wire [31:0] trial = {root, 2'b01};

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st   <= ST_IDLE;
      i    <= 4'd0;
      x_r  <= 32'd0;
      rem  <= 32'd0;
      root <= 16'd0;
      y    <= 16'd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            x_r  <= x;
            rem  <= 32'd0;
            root <= 16'd0;
            i    <= 4'd0;
            st   <= ST_RUN;
          end
        end
        ST_RUN: begin
          x_r <= {x_r[29:0], 2'b00};
          if (rem_n >= trial) begin
            rem  <= rem_n - trial;
            root <= {root[14:0], 1'b1};
          end else begin
            rem  <= rem_n;
            root <= {root[14:0], 1'b0};
          end
          if (i == 4'd15) begin
            y    <= (rem_n >= trial) ? {root[14:0], 1'b1} : {root[14:0], 1'b0};
            done <= 1'b1;
            st   <= ST_IDLE;
          end
          i <= i + 1'b1;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
