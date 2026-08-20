// One-head decode attention MAC skeleton (no softmax).
// scores[t] = (k[t] · q) >>> SHIFT
// o         = (sum_t scores[t] * v[t]) >>> SHIFT
// D lanes multiply in parallel → 2*S + 2 cycles (QK sweep + AV sweep + pack).
`timescale 1ns / 1ps

module attn_decode #(
  parameter integer D     = 4,
  parameter integer S_MAX = 64,
  parameter integer QK_W  = 8,
  parameter integer V_W   = 8,
  parameter integer SHIFT = 8,
  parameter integer O_W   = 24
) (
  input  wire                         clk,
  input  wire                         rst_n,
  input  wire                         en,
  input  wire [7:0]                   seq_len,
  input  wire signed [D*QK_W-1:0]     q_flat,
  input  wire signed [S_MAX*D*QK_W-1:0] k_flat,
  input  wire signed [S_MAX*D*V_W-1:0]  v_flat,
  output reg  signed [D*O_W-1:0]      o_flat,
  output reg                          done
);
  localparam integer IDX_W = (S_MAX <= 2) ? 1 : $clog2(S_MAX);
  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_QK   = 2'd1;
  localparam [1:0] ST_AV   = 2'd2;
  localparam [1:0] ST_PACK = 2'd3;

  wire signed [QK_W-1:0] q_i [0:D-1];
  genvar u;
  generate
    for (u = 0; u < D; u = u + 1) begin : uq
      assign q_i[u] = q_flat[u*QK_W +: QK_W];
    end
  endgenerate

  function signed [QK_W-1:0] k_at;
    input integer t;
    input integer d;
    begin
      k_at = k_flat[(t*D+d)*QK_W +: QK_W];
    end
  endfunction
  function signed [V_W-1:0] v_at;
    input integer t;
    input integer d;
    begin
      v_at = v_flat[(t*D+d)*V_W +: V_W];
    end
  endfunction

  reg [1:0] st;
  reg [IDX_W-1:0] t;
  reg [7:0] S;
  reg signed [QK_W-1:0] q_r [0:D-1];
  reg signed [31:0] score [0:S_MAX-1];
  reg signed [31:0] acc [0:D-1];
  integer ii;

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      t  <= {IDX_W{1'b0}};
      S  <= 8'd1;
      o_flat <= {D*O_W{1'b0}};
      for (ii = 0; ii < D; ii = ii + 1) begin
        q_r[ii] <= {QK_W{1'b0}};
        acc[ii] <= 32'sd0;
      end
      for (ii = 0; ii < S_MAX; ii = ii + 1)
        score[ii] <= 32'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            S <= (seq_len == 8'd0) ? 8'd1 : seq_len;
            for (ii = 0; ii < D; ii = ii + 1)
              q_r[ii] <= q_i[ii];
            t  <= {IDX_W{1'b0}};
            st <= ST_QK;
          end
        end
        ST_QK: begin
          begin : qk_dot
            reg signed [31:0] s;
            s = 32'sd0;
            for (ii = 0; ii < D; ii = ii + 1)
              s = s + q_r[ii] * k_at(t, ii);
            score[t] <= s >>> SHIFT;
          end
          if ({24'd0, t} + 1 >= S) begin
            t  <= {IDX_W{1'b0}};
            for (ii = 0; ii < D; ii = ii + 1)
              acc[ii] <= 32'sd0;
            st <= ST_AV;
          end else
            t <= t + 1'b1;
        end
        ST_AV: begin
          for (ii = 0; ii < D; ii = ii + 1)
            acc[ii] <= acc[ii] + score[t] * v_at(t, ii);
          if ({24'd0, t} + 1 >= S) begin
            st <= ST_PACK;
          end else
            t <= t + 1'b1;
        end
        ST_PACK: begin
          for (ii = 0; ii < D; ii = ii + 1)
            o_flat[ii*O_W +: O_W] <= acc[ii] >>> SHIFT;
          done <= 1'b1;
          st   <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
