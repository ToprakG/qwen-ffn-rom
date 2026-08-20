// One DeltaNet decoder layer (toy width): hidden=8, mixer D=4.
// rms1 → mixer → residual → rms2 → 8x8 FFN tap → residual.
// Bit-exact vs quant/layer_int.py decoder_layer_int.
`timescale 1ns / 1ps

module decoder_layer (
  input  wire                clk,
  input  wire                rst_n,
  input  wire                en,
  input  wire signed [63:0]  x_flat,
  input  wire        [7:0]   g,
  input  wire        [7:0]   beta,
  input  wire signed [63:0]  w_n1_flat,
  input  wire signed [63:0]  w_n2_flat,
  output reg  signed [63:0]  y_flat,
  output reg                 done,
  output wire                ready
);
  localparam [3:0] ST_IDLE  = 4'd0;
  localparam [3:0] ST_N1E   = 4'd1;
  localparam [3:0] ST_N1    = 4'd2;
  localparam [3:0] ST_MIXE  = 4'd3;
  localparam [3:0] ST_MIX   = 4'd4;
  localparam [3:0] ST_RES1  = 4'd5;
  localparam [3:0] ST_N2E   = 4'd6;
  localparam [3:0] ST_N2    = 4'd7;
  localparam [3:0] ST_FFNE  = 4'd8;
  localparam [3:0] ST_FFN   = 4'd9;
  localparam [3:0] ST_OUT   = 4'd10;

  integer k;
  reg [3:0] st;
  reg signed [7:0] x_r [0:7];
  reg signed [7:0] mid [0:7];
  reg [7:0] g_r;
  reg [7:0] beta_r;

  function automatic signed [63:0] pack8;
    input signed [7:0] a0, a1, a2, a3, a4, a5, a6, a7;
    begin
      pack8 = {a7, a6, a5, a4, a3, a2, a1, a0};
    end
  endfunction

  wire signed [63:0] x_now = pack8(x_r[0], x_r[1], x_r[2], x_r[3], x_r[4], x_r[5], x_r[6], x_r[7]);
  wire signed [63:0] mid_now = pack8(mid[0], mid[1], mid[2], mid[3], mid[4], mid[5], mid[6], mid[7]);

  reg n1_en, n2_en, mix_en, ffn_en;
  wire n1_done, n2_done, mix_done, mix_ready, ffn_done;
  wire signed [63:0] h1, h2;
  wire signed [4*24-1:0] o_mix;
  wire signed [119:0] y_ffn;

  rmsnorm8 u_n1 (
    .clk(clk), .rst_n(rst_n), .en(n1_en),
    .x_flat(x_now), .w_flat(w_n1_flat), .y_flat(h1), .done(n1_done)
  );
  rmsnorm8 u_n2 (
    .clk(clk), .rst_n(rst_n), .en(n2_en),
    .x_flat(mid_now), .w_flat(w_n2_flat), .y_flat(h2), .done(n2_done)
  );

  wire signed [31:0] qkv = h1[31:0];
  gated_delta_d4_bram u_mix (
    .clk(clk), .rst_n(rst_n), .en(mix_en),
    .q_flat(qkv), .k_flat(qkv), .v_flat(qkv),
    .g(g_r), .beta(beta_r),
    .o_flat(o_mix), .done(mix_done), .ready(mix_ready)
  );

  ffn_tap_unit u_ffn (
    .clk(clk), .rst_n(rst_n), .en(ffn_en),
    .x_flat(h2), .y_flat(y_ffn), .done(ffn_done)
  );

  assign ready = (st == ST_IDLE) && mix_ready;

  wire signed [23:0] o_lane [0:3];
  wire signed [14:0] y_lane [0:7];
  genvar gi;
  generate
    for (gi = 0; gi < 4; gi = gi + 1) begin : om
      assign o_lane[gi] = o_mix[gi*24 +: 24];
    end
    for (gi = 0; gi < 8; gi = gi + 1) begin : ym
      assign y_lane[gi] = y_ffn[gi*15 +: 15];
    end
  endgenerate

  wire signed [31:0] res1 [0:3];
  wire signed [7:0]  mid4 [0:3];
  generate
    for (gi = 0; gi < 4; gi = gi + 1) begin : r1
      assign res1[gi] = x_r[gi] + (o_lane[gi] >>> 8);
      assign mid4[gi] = (res1[gi] > 32'sd127) ? 8'sd127 :
                        (res1[gi] < -32'sd128) ? -8'sd128 : res1[gi][7:0];
    end
  endgenerate

  wire signed [31:0] res2 [0:7];
  wire signed [7:0]  ysat [0:7];
  generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : r2
      assign res2[gi] = mid[gi] + (y_lane[gi] >>> 7);
      assign ysat[gi] = (res2[gi] > 32'sd127) ? 8'sd127 :
                        (res2[gi] < -32'sd128) ? -8'sd128 : res2[gi][7:0];
    end
  endgenerate

  always @(posedge clk) begin
    done   <= 1'b0;
    n1_en  <= 1'b0;
    n2_en  <= 1'b0;
    mix_en <= 1'b0;
    ffn_en <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      g_r <= 8'd0;
      beta_r <= 8'd0;
      y_flat <= 64'sd0;
      for (k = 0; k < 8; k = k + 1) begin
        x_r[k] <= 8'sd0;
        mid[k] <= 8'sd0;
      end
    end else begin
      case (st)
        ST_IDLE: begin
          if (en && mix_ready) begin
            for (k = 0; k < 8; k = k + 1)
              x_r[k] <= x_flat[k*8 +: 8];
            g_r    <= g;
            beta_r <= beta;
            st     <= ST_N1E;
          end
        end
        ST_N1E: begin
          n1_en <= 1'b1;
          st    <= ST_N1;
        end
        ST_N1: begin
          if (n1_done)
            st <= ST_MIXE;
        end
        ST_MIXE: begin
          mix_en <= 1'b1;
          st     <= ST_MIX;
        end
        ST_MIX: begin
          if (mix_done)
            st <= ST_RES1;
        end
        ST_RES1: begin
          for (k = 0; k < 4; k = k + 1)
            mid[k] <= mid4[k];
          for (k = 4; k < 8; k = k + 1)
            mid[k] <= x_r[k];
          st <= ST_N2E;
        end
        ST_N2E: begin
          n2_en <= 1'b1;
          st    <= ST_N2;
        end
        ST_N2: begin
          if (n2_done)
            st <= ST_FFNE;
        end
        ST_FFNE: begin
          ffn_en <= 1'b1;
          st     <= ST_FFN;
        end
        ST_FFN: begin
          if (ffn_done)
            st <= ST_OUT;
        end
        ST_OUT: begin
          for (k = 0; k < 8; k = k + 1)
            y_flat[k*8 +: 8] <= ysat[k];
          done <= 1'b1;
          st   <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
