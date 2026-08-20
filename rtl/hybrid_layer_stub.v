// One-layer decode stub: Gated DeltaNet D=4 + winning FFN column (8x8 CSD).
// Shared int8 x. q=k=v are x[0:3] (no QKV projections). FFN sees the same x
// in parallel with the mixer; y is sampled when the mixer pulses done.
// Layer cycles/token track the mixer (~69), not the 1-cycle FFN.
`timescale 1ns / 1ps

module hybrid_layer_stub (
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 en,
  input  wire signed [63:0]   x_flat,
  input  wire        [7:0]    g,
  input  wire        [7:0]    beta,
  output reg  signed [95:0]   o_delta_flat,
  output reg  signed [119:0]  y_ffn_flat,
  output reg                  done
);
  localparam [0:0] ST_IDLE = 1'd0;
  localparam [0:0] ST_MIX  = 1'd1;

  reg st;
  reg signed [63:0] x_r;
  reg [7:0] g_r;
  reg [7:0] beta_r;

  wire take = (st == ST_IDLE) && en;
  wire signed [31:0] qkv = take ? x_flat[31:0] : x_r[31:0];
  wire [7:0] g_now = take ? g : g_r;
  wire [7:0] beta_now = take ? beta : beta_r;

  wire signed [95:0]  d_o;
  wire                d_done;
  wire signed [119:0] y_c;

  gated_delta_step u_delta (
    .clk(clk),
    .rst_n(rst_n),
    .en(take),
    .q_flat(qkv),
    .k_flat(qkv),
    .v_flat(qkv),
    .g(g_now),
    .beta(beta_now),
    .o_flat(d_o),
    .done(d_done)
  );

  ffn_tile u_ffn (
    .x_flat(x_r),
    .y_flat(y_c)
  );

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      x_r <= 64'sd0;
      g_r <= 8'd0;
      beta_r <= 8'd0;
      o_delta_flat <= 96'sd0;
      y_ffn_flat <= 120'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            x_r <= x_flat;
            g_r <= g;
            beta_r <= beta;
            st <= ST_MIX;
          end
        end
        ST_MIX: begin
          if (d_done) begin
            o_delta_flat <= d_o;
            y_ffn_flat <= y_c;
            done <= 1'b1;
            st <= ST_IDLE;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
