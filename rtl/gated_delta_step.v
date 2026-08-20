// Gated DeltaNet PE — integer recurrence, one MAC, D-parameter FSM.
// Same math as quant/delta_int.py. Work is O(D^2) per token, not O(seq).
// D=4 is the signed-off PE. D=16 is the step-7 size check. Do not unroll.
`timescale 1ns / 1ps

module gated_delta_step #(
  parameter integer D     = 4,
  parameter integer QK_W  = 8,
  parameter integer V_W   = 8,
  parameter integer S_W   = 16,
  parameter integer G_W   = 8,
  parameter integer SHIFT = 8,
  parameter integer O_W   = 24
) (
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     en,
  input  wire signed [D*QK_W-1:0] q_flat,
  input  wire signed [D*QK_W-1:0] k_flat,
  input  wire signed [D*V_W-1:0]  v_flat,
  input  wire        [G_W-1:0]    g,
  input  wire        [G_W-1:0]    beta,
  output reg  signed [D*O_W-1:0]  o_flat,
  output reg                      done
);
  localparam integer IDX_W = (D <= 2) ? 1 : $clog2(D);
  localparam [IDX_W-1:0] LAST = D - 1;
  localparam signed [31:0] S_MAX = 32'sd32767;
  localparam signed [31:0] S_MIN = -32'sd32768;

  localparam [2:0] ST_IDLE  = 3'd0;
  localparam [2:0] ST_DECAY = 3'd1;
  localparam [2:0] ST_KV    = 3'd2;
  localparam [2:0] ST_DELTA = 3'd3;
  localparam [2:0] ST_OUTER = 3'd4;
  localparam [2:0] ST_OUT   = 3'd5;
  localparam [2:0] ST_PACK  = 3'd6;

  wire signed [QK_W-1:0] q_i [0:D-1];
  wire signed [QK_W-1:0] k_i [0:D-1];
  wire signed [V_W-1:0]  v_i [0:D-1];
  genvar u;
  generate
    for (u = 0; u < D; u = u + 1) begin : unpack
      assign q_i[u] = q_flat[u*QK_W +: QK_W];
      assign k_i[u] = k_flat[u*QK_W +: QK_W];
      assign v_i[u] = v_flat[u*V_W +: V_W];
    end
  endgenerate

  reg signed [QK_W-1:0] q_r [0:D-1];
  reg signed [QK_W-1:0] k_r [0:D-1];
  reg signed [V_W-1:0]  v_r [0:D-1];
  reg signed [G_W:0]    g_r;
  reg signed [G_W:0]    beta_r;

  reg signed [S_W-1:0] S [0:D-1][0:D-1];
  reg signed [31:0]    kv [0:D-1];
  reg signed [31:0]    delta [0:D-1];
  reg signed [O_W-1:0] o_r [0:D-1];

  reg [2:0] st;
  reg [IDX_W-1:0] i;
  reg [IDX_W-1:0] j;
  reg signed [31:0] acc;

  wire signed [31:0] s_ij     = {{(32-S_W){S[i][j][S_W-1]}}, S[i][j]};
  wire signed [31:0] decay_p  = S[i][j] * g_r;
  wire signed [31:0] kv_p     = S[i][j] * k_r[i];
  wire signed [31:0] v_ext    = {{(32-V_W){v_r[j][V_W-1]}}, v_r[j]};
  wire signed [31:0] delta_p  = beta_r * (v_ext - kv[j]);
  wire signed [31:0] outer_p  = k_r[i] * delta[j];
  wire signed [31:0] outer_s  = s_ij + (outer_p >>> SHIFT);
  wire signed [S_W-1:0] s_sat = (outer_s > S_MAX) ? S_MAX[S_W-1:0] :
                                (outer_s < S_MIN) ? S_MIN[S_W-1:0] :
                                outer_s[S_W-1:0];
  wire signed [31:0] out_p    = S[i][j] * q_r[i];

  integer ii, jj;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      i <= {IDX_W{1'b0}};
      j <= {IDX_W{1'b0}};
      acc <= 32'sd0;
      o_flat <= {D * O_W{1'b0}};
      for (ii = 0; ii < D; ii = ii + 1) begin
        kv[ii] <= 32'sd0;
        delta[ii] <= 32'sd0;
        o_r[ii] <= {O_W{1'b0}};
        q_r[ii] <= {QK_W{1'b0}};
        k_r[ii] <= {QK_W{1'b0}};
        v_r[ii] <= {V_W{1'b0}};
        for (jj = 0; jj < D; jj = jj + 1)
          S[ii][jj] <= {S_W{1'b0}};
      end
      g_r <= {(G_W+1){1'b0}};
      beta_r <= {(G_W+1){1'b0}};
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (ii = 0; ii < D; ii = ii + 1) begin
              q_r[ii] <= q_i[ii];
              k_r[ii] <= k_i[ii];
              v_r[ii] <= v_i[ii];
            end
            g_r    <= {1'b0, g};
            beta_r <= {1'b0, beta};
            i <= {IDX_W{1'b0}};
            j <= {IDX_W{1'b0}};
            st <= ST_DECAY;
          end
        end

        ST_DECAY: begin
          S[i][j] <= decay_p >>> SHIFT;
          if (j == LAST) begin
            j <= {IDX_W{1'b0}};
            if (i == LAST) begin
              i <= {IDX_W{1'b0}};
              acc <= 32'sd0;
              st <= ST_KV;
            end else
              i <= i + 1'b1;
          end else
            j <= j + 1'b1;
        end

        ST_KV: begin
          if (i == {IDX_W{1'b0}}) begin
            acc <= kv_p;
            i <= i + 1'b1;
          end else if (i == LAST) begin
            kv[j] <= (acc + kv_p) >>> SHIFT;
            i <= {IDX_W{1'b0}};
            if (j == LAST) begin
              j <= {IDX_W{1'b0}};
              st <= ST_DELTA;
            end else
              j <= j + 1'b1;
          end else begin
            acc <= acc + kv_p;
            i <= i + 1'b1;
          end
        end

        ST_DELTA: begin
          delta[j] <= delta_p >>> SHIFT;
          if (j == LAST) begin
            j <= {IDX_W{1'b0}};
            i <= {IDX_W{1'b0}};
            st <= ST_OUTER;
          end else
            j <= j + 1'b1;
        end

        ST_OUTER: begin
          S[i][j] <= s_sat;
          if (j == LAST) begin
            j <= {IDX_W{1'b0}};
            if (i == LAST) begin
              i <= {IDX_W{1'b0}};
              acc <= 32'sd0;
              st <= ST_OUT;
            end else
              i <= i + 1'b1;
          end else
            j <= j + 1'b1;
        end

        ST_OUT: begin
          if (i == {IDX_W{1'b0}}) begin
            acc <= out_p;
            i <= i + 1'b1;
          end else if (i == LAST) begin
            o_r[j] <= (acc + out_p) >>> SHIFT;
            i <= {IDX_W{1'b0}};
            if (j == LAST) begin
              j <= {IDX_W{1'b0}};
              st <= ST_PACK;
            end else
              j <= j + 1'b1;
          end else begin
            acc <= acc + out_p;
            i <= i + 1'b1;
          end
        end

        ST_PACK: begin
          for (ii = 0; ii < D; ii = ii + 1)
            o_flat[ii*O_W +: O_W] <= o_r[ii];
          done <= 1'b1;
          st <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
