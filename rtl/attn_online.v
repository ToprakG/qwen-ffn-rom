// Fused online-softmax decode attention.
//
// One pass over KV: each cycle reads P banked (k,v) rows, updates running
// (m, ℓ, o) for every Q head. No S-length score vector.
//
// Banking: position t lives in bank (t % P) at address (t / P). Same 1R1W
// inferred-BRAM discipline as rtl/delta_s_col_ram.v. P must be a power of 2.
//
// GQA: N_KV key/value heads; each is reused across N_Q_PER query heads.
// Cycles = ceil(S/P)+2, independent of N_KV and N_Q_PER (they overlap).
//
// int4 KV in the banks; q is int8. Bit-exact vs quant/attn_online_int.py.
`timescale 1ns / 1ps

module attn_online #(
  parameter integer D       = 8,
  parameter integer P       = 4,
  parameter integer S_MAX   = 64,
  parameter integer N_KV    = 1,
  parameter integer N_Q_PER = 1,
  parameter integer KV_W    = 4,
  parameter integer QK_W    = 8,
  parameter integer SHIFT   = 8,
  parameter integer O_W     = 24,
  parameter integer N_Q     = N_KV * N_Q_PER
) (
  input  wire                              clk,
  input  wire                              rst_n,
  input  wire                              en,
  input  wire [15:0]                       seq_len,
  input  wire signed [N_Q*D*QK_W-1:0]      q_flat,
  input  wire                              wr_en,
  input  wire [15:0]                       wr_t,
  input  wire [N_KV*D*KV_W-1:0]            wr_k,
  input  wire [N_KV*D*KV_W-1:0]            wr_v,
  output reg  signed [N_Q*D*O_W-1:0]       o_flat,
  output reg                               done,
  output wire                              ready
);
  localparam integer P_W     = (P <= 2) ? 1 : $clog2(P);
  localparam integer DEPTH   = S_MAX / P;
  localparam integer ADDR_W  = (DEPTH <= 2) ? 1 : $clog2(DEPTH);
  localparam integer WIDTH_K = D * KV_W;
  localparam integer EXP_Q   = 8;
  localparam integer EXP_MAX = 32;
  localparam [1:0] ST_IDLE   = 2'd0;
  localparam [1:0] ST_ISS    = 2'd1;
  localparam [1:0] ST_EX     = 2'd2;

  genvar hq, dq, gkv, gp;

  wire signed [QK_W-1:0] q_i [0:N_Q-1][0:D-1];
  generate
    for (hq = 0; hq < N_Q; hq = hq + 1) begin : uq
      for (dq = 0; dq < D; dq = dq + 1) begin : ud
        assign q_i[hq][dq] = q_flat[(hq*D+dq)*QK_W +: QK_W];
      end
    end
  endgenerate

  wire [P_W-1:0]    wr_bank = wr_t[P_W-1:0];
  wire [ADDR_W-1:0] wr_addr = wr_t[15:P_W];

  wire [WIDTH_K-1:0] rdata_k [0:N_KV-1][0:P-1];
  wire [WIDTH_K-1:0] rdata_v [0:N_KV-1][0:P-1];

  reg [ADDR_W-1:0] step_iss;
  reg [ADDR_W-1:0] step_ex;
  reg              ex_v;
  reg [1:0]        st;

  generate
    for (gkv = 0; gkv < N_KV; gkv = gkv + 1) begin : g_kv
      for (gp = 0; gp < P; gp = gp + 1) begin : g_bank
        wire we = wr_en && (wr_bank == gp[P_W-1:0]) && ready;
        kv_seq_ram #(
          .DEPTH(DEPTH), .WIDTH(WIDTH_K), .ADDR_W(ADDR_W)
        ) u_k (
          .clk(clk), .we(we), .waddr(wr_addr),
          .wdata(wr_k[gkv*WIDTH_K +: WIDTH_K]),
          .raddr(step_iss), .rdata(rdata_k[gkv][gp])
        );
        kv_seq_ram #(
          .DEPTH(DEPTH), .WIDTH(WIDTH_K), .ADDR_W(ADDR_W)
        ) u_v (
          .clk(clk), .we(we), .waddr(wr_addr),
          .wdata(wr_v[gkv*WIDTH_K +: WIDTH_K]),
          .raddr(step_iss), .rdata(rdata_v[gkv][gp])
        );
      end
    end
  endgenerate

  function signed [31:0] tdiv;
    input signed [31:0] n;
    input signed [31:0] d;
    reg [31:0] ad;
    reg [31:0] an;
    begin
      ad = d[31] ? (~d + 1'b1) : d;
      if (ad == 32'd0)
        ad = 32'd1;
      an = n[31] ? (~n + 1'b1) : n;
      tdiv = n[31] ? -$signed(an / ad) : $signed(an / ad);
    end
  endfunction

  reg signed [QK_W-1:0] q_r   [0:N_Q-1][0:D-1];
  reg [15:0]            S;
  reg [ADDR_W-1:0]      last_iss;
  reg                   have  [0:N_Q-1];
  reg signed [31:0]     m_r   [0:N_Q-1];
  reg signed [31:0]     ell_r [0:N_Q-1];
  reg signed [31:0]     o_r   [0:N_Q-1][0:D-1];

  wire [8:0] lut_w [0:N_Q-1][0:P-1];
  wire [8:0] lut_s [0:N_Q-1];

  reg signed [31:0] score [0:N_Q-1][0:P-1];
  reg signed [31:0] m_blk [0:N_Q-1];
  reg signed [31:0] m_new [0:N_Q-1];
  reg [8:0]         scale [0:N_Q-1];
  reg signed [31:0] ell_n [0:N_Q-1];
  reg signed [31:0] o_n   [0:N_Q-1][0:D-1];

  integer h, p, d, tpos, kv, acc, wacc;
  integer ii, jj;
  reg signed [3:0] k4;
  reg signed [3:0] v4;
  reg signed [31:0] qe, ke;

  always @* begin
    acc = 0;
    k4  = 4'sd0;
    qe  = 32'sd0;
    ke  = 32'sd0;
    for (h = 0; h < N_Q; h = h + 1) begin
      kv = h / N_Q_PER;
      for (p = 0; p < P; p = p + 1) begin
        tpos = step_ex * P + p;
        if (tpos >= S) begin
          score[h][p] = 32'sh80000000;
        end else begin
          acc = 0;
          for (d = 0; d < D; d = d + 1) begin
            k4 = $signed(rdata_k[kv][p][d*KV_W +: KV_W]);
            qe = {{(32-QK_W){q_r[h][d][QK_W-1]}}, q_r[h][d]};
            ke = {{28{k4[3]}}, k4};
            acc = acc + qe * ke;
          end
          score[h][p] = acc >>> SHIFT;
        end
      end
      m_blk[h] = score[h][0];
      for (p = 1; p < P; p = p + 1)
        if (score[h][p] > m_blk[h])
          m_blk[h] = score[h][p];
      if (!have[h])
        m_new[h] = m_blk[h];
      else
        m_new[h] = (m_blk[h] > m_r[h]) ? m_blk[h] : m_r[h];
    end
  end

  generate
    for (hq = 0; hq < N_Q; hq = hq + 1) begin : g_lut
      wire signed [31:0] dlt = (m_new[hq] > m_r[hq]) ? (m_new[hq] - m_r[hq]) : 32'sd0;
      wire [5:0] dlt_u = (dlt > EXP_MAX) ? EXP_MAX[5:0] : dlt[5:0];
      attn_exp_lut u_sc (.d(dlt_u), .y(lut_s[hq]));
      for (gp = 0; gp < P; gp = gp + 1) begin : g_w
        wire signed [31:0] wd = m_new[hq] - score[hq][gp];
        wire [5:0] wd_u = (wd < 0) ? 6'd0 : ((wd > EXP_MAX) ? EXP_MAX[5:0] : wd[5:0]);
        attn_exp_lut u_w (.d(wd_u), .y(lut_w[hq][gp]));
      end
    end
  endgenerate

  always @* begin
    acc  = 0;
    wacc = 0;
    v4   = 4'sd0;
    for (h = 0; h < N_Q; h = h + 1) begin
      kv = h / N_Q_PER;
      scale[h] = have[h] ? lut_s[h] : 9'd0;
      wacc = 0;
      for (p = 0; p < P; p = p + 1) begin
        tpos = step_ex * P + p;
        if (tpos < S)
          wacc = wacc + lut_w[h][p];
      end
      ell_n[h] = (ell_r[h] * $signed({23'd0, scale[h]})) >>> EXP_Q;
      ell_n[h] = ell_n[h] + wacc;
      for (d = 0; d < D; d = d + 1) begin
        acc = (o_r[h][d] * $signed({23'd0, scale[h]})) >>> EXP_Q;
        for (p = 0; p < P; p = p + 1) begin
          tpos = step_ex * P + p;
          if (tpos < S) begin
            v4 = $signed(rdata_v[kv][p][d*KV_W +: KV_W]);
            acc = acc + $signed({23'd0, lut_w[h][p]}) * {{28{v4[3]}}, v4};
          end
        end
        o_n[h][d] = acc;
      end
    end
  end

  assign ready = (st == ST_IDLE);

  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st       <= ST_IDLE;
      step_iss <= {ADDR_W{1'b0}};
      step_ex  <= {ADDR_W{1'b0}};
      ex_v     <= 1'b0;
      S        <= 16'd1;
      last_iss <= {ADDR_W{1'b0}};
      o_flat   <= {N_Q*D*O_W{1'b0}};
      for (ii = 0; ii < N_Q; ii = ii + 1) begin
        have[ii]  <= 1'b0;
        m_r[ii]   <= 32'sd0;
        ell_r[ii] <= 32'sd0;
        for (jj = 0; jj < D; jj = jj + 1) begin
          q_r[ii][jj] <= {QK_W{1'b0}};
          o_r[ii][jj] <= 32'sd0;
        end
      end
    end else begin
      case (st)
        ST_IDLE: begin
          ex_v <= 1'b0;
          if (en) begin
            begin : cap
              integer s_cap, n_steps;
              s_cap = (seq_len == 16'd0) ? 1 :
                      (seq_len > S_MAX[15:0]) ? S_MAX : seq_len;
              n_steps = (s_cap + P - 1) >> P_W;
              S        <= s_cap[15:0];
              last_iss <= (n_steps <= 1) ? {ADDR_W{1'b0}} : (n_steps - 1);
            end
            step_iss <= {ADDR_W{1'b0}};
            for (ii = 0; ii < N_Q; ii = ii + 1) begin
              have[ii]  <= 1'b0;
              m_r[ii]   <= 32'sd0;
              ell_r[ii] <= 32'sd0;
              for (jj = 0; jj < D; jj = jj + 1) begin
                q_r[ii][jj] <= q_i[ii][jj];
                o_r[ii][jj] <= 32'sd0;
              end
            end
            st <= ST_ISS;
          end
        end

        ST_ISS: begin
          step_ex <= step_iss;
          ex_v    <= 1'b1;
          if (step_iss == last_iss)
            st <= ST_EX;
          else
            step_iss <= step_iss + 1'b1;
        end

        ST_EX: begin
          ex_v <= 1'b0;
          for (ii = 0; ii < N_Q; ii = ii + 1)
            for (jj = 0; jj < D; jj = jj + 1)
              o_flat[(ii*D+jj)*O_W +: O_W] <= tdiv(o_n[ii][jj] * 32'sd256, ell_n[ii]);
          done <= 1'b1;
          st   <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase

      if ((st == ST_ISS && ex_v) || (st == ST_EX)) begin
        for (ii = 0; ii < N_Q; ii = ii + 1) begin
          have[ii]  <= 1'b1;
          m_r[ii]   <= m_new[ii];
          ell_r[ii] <= ell_n[ii];
          for (jj = 0; jj < D; jj = jj + 1)
            o_r[ii][jj] <= o_n[ii][jj];
        end
      end
    end
  end
endmodule
