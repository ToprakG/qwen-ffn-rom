// 32-cycle unsigned divide: q = num / den (32-bit / 16-bit → 16-bit quotient).
`timescale 1ns / 1ps

module idiv_u32 (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        en,
  input  wire [31:0] num,
  input  wire [15:0] den,
  output reg  [31:0] q,
  output reg         done
);
  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_RUN  = 2'd1;

  reg [1:0]  st;
  reg [5:0]  i;
  reg [31:0] n_r;
  reg [31:0] r;
  reg [31:0] qq;
  reg [15:0] d;
  wire [31:0] r_n = {r[30:0], n_r[31]};

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st  <= ST_IDLE;
      i   <= 6'd0;
      n_r <= 32'd0;
      r   <= 32'd0;
      qq  <= 32'd0;
      d   <= 16'd1;
      q   <= 32'd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            n_r <= num;
            r   <= 32'd0;
            qq  <= 32'd0;
            d   <= (den == 16'd0) ? 16'd1 : den;
            i   <= 6'd0;
            st  <= ST_RUN;
          end
        end
        ST_RUN: begin
          n_r <= {n_r[30:0], 1'b0};
          if (r_n >= {16'd0, d}) begin
            r  <= r_n - {16'd0, d};
            qq <= {qq[30:0], 1'b1};
          end else begin
            r  <= r_n;
            qq <= {qq[30:0], 1'b0};
          end
          if (i == 6'd31) begin
            q    <= (r_n >= {16'd0, d}) ? {qq[30:0], 1'b1} : {qq[30:0], 1'b0};
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
