// Gated DeltaNet PE — D parallel MACs for row/column work.
// Same integer math as quant/delta_int.py and rtl/gated_delta_step.v (1-MAC).
// Inner products (kᵀS, qᵀS) are one D-wide MAC + adder tree per cycle.
// Decay / outer are one row (D lanes) per cycle. Do not unroll D×D.
// D must be a power of two. D=16 is the step-7 parallel PE.
`timescale 1ns / 1ps

module delta_dot_tree #(
  parameter integer N = 16,
  parameter integer W = 32
) (
  input  wire signed [N*W-1:0] xs,
  output reg  signed [W-1:0]   y
);
  integer k;
  reg signed [W-1:0] acc;
  always @* begin
    acc = {W{1'b0}};
    for (k = 0; k < N; k = k + 1)
      acc = acc + $signed(xs[k*W +: W]);
    y = acc;
  end
endmodule

module gated_delta_dpar #(
  parameter integer D     = 16,
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

  wire signed [31:0] prod [0:D-1];
  wire signed [S_W-1:0] s_sat [0:D-1];
  wire signed [D*32-1:0] prod_flat;
  wire signed [31:0] dot;

  genvar t;
  generate
    for (t = 0; t < D; t = t + 1) begin : lanes
      reg signed [31:0] mul_a;
      reg signed [31:0] mul_b;
      wire signed [31:0] s_row = {{(32-S_W){S[i][t][S_W-1]}}, S[i][t]};
      wire signed [31:0] s_col = {{(32-S_W){S[t][j][S_W-1]}}, S[t][j]};
      wire signed [31:0] v_ext = {{(32-V_W){v_r[t][V_W-1]}}, v_r[t]};
      wire signed [31:0] outer_s = s_row + (prod[t] >>> SHIFT);

      always @* begin
        mul_a = 32'sd0;
        mul_b = 32'sd0;
        case (st)
          ST_DECAY: begin
            mul_a = s_row;
            mul_b = {{(32-(G_W+1)){g_r[G_W]}}, g_r};
          end
          ST_KV: begin
            mul_a = s_col;
            mul_b = {{(32-QK_W){k_r[t][QK_W-1]}}, k_r[t]};
          end
          ST_DELTA: begin
            mul_a = v_ext - kv[t];
            mul_b = {{(32-(G_W+1)){beta_r[G_W]}}, beta_r};
          end
          ST_OUTER: begin
            mul_a = {{(32-QK_W){k_r[i][QK_W-1]}}, k_r[i]};
            mul_b = delta[t];
          end
          ST_OUT: begin
            mul_a = s_col;
            mul_b = {{(32-QK_W){q_r[t][QK_W-1]}}, q_r[t]};
          end
          default: begin
            mul_a = 32'sd0;
            mul_b = 32'sd0;
          end
        endcase
      end

      assign prod[t] = mul_a * mul_b;
      assign prod_flat[t*32 +: 32] = prod[t];
      assign s_sat[t] = (outer_s > S_MAX) ? S_MAX[S_W-1:0] :
                        (outer_s < S_MIN) ? S_MIN[S_W-1:0] :
                        outer_s[S_W-1:0];
    end
  endgenerate

  delta_dot_tree #(.N(D), .W(32)) u_dot (
    .xs(prod_flat),
    .y(dot)
  );

  integer ii, tt;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      i <= {IDX_W{1'b0}};
      j <= {IDX_W{1'b0}};
      o_flat <= {D * O_W{1'b0}};
      for (ii = 0; ii < D; ii = ii + 1) begin
        kv[ii] <= 32'sd0;
        delta[ii] <= 32'sd0;
        o_r[ii] <= {O_W{1'b0}};
        q_r[ii] <= {QK_W{1'b0}};
        k_r[ii] <= {QK_W{1'b0}};
        v_r[ii] <= {V_W{1'b0}};
        for (tt = 0; tt < D; tt = tt + 1)
          S[ii][tt] <= {S_W{1'b0}};
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
          for (tt = 0; tt < D; tt = tt + 1)
            S[i][tt] <= prod[tt] >>> SHIFT;
          if (i == LAST) begin
            i <= {IDX_W{1'b0}};
            j <= {IDX_W{1'b0}};
            st <= ST_KV;
          end else
            i <= i + 1'b1;
        end

        ST_KV: begin
          kv[j] <= dot >>> SHIFT;
          if (j == LAST) begin
            j <= {IDX_W{1'b0}};
            st <= ST_DELTA;
          end else
            j <= j + 1'b1;
        end

        ST_DELTA: begin
          for (tt = 0; tt < D; tt = tt + 1)
            delta[tt] <= prod[tt] >>> SHIFT;
          i <= {IDX_W{1'b0}};
          j <= {IDX_W{1'b0}};
          st <= ST_OUTER;
        end

        ST_OUTER: begin
          for (tt = 0; tt < D; tt = tt + 1)
            S[i][tt] <= s_sat[tt];
          if (i == LAST) begin
            i <= {IDX_W{1'b0}};
            j <= {IDX_W{1'b0}};
            st <= ST_OUT;
          end else
            i <= i + 1'b1;
        end

        ST_OUT: begin
          o_r[j] <= dot >>> SHIFT;
          if (j == LAST) begin
            j <= {IDX_W{1'b0}};
            st <= ST_PACK;
          end else
            j <= j + 1'b1;
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
