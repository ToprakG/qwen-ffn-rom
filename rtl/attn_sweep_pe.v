// 27B attention sweep PE: D=256, P=4 KV-SRAM banks, 1 Q head.
// Generate-based D-dot (not a 256-ripple always-loop). Cycles = ceil(S/P)+2.
// Combo tdiv on the last cycle is the same as rtl/attn_online.v.
`timescale 1ns / 1ps

module attn_dot_i8i4 #(
  parameter integer D = 256
) (
  input  wire signed [D*8-1:0] q_flat,
  input  wire        [D*4-1:0] k_flat,
  output wire signed [31:0]    score
);
  wire signed [D*32-1:0] prods;
  genvar d;
  generate
    for (d = 0; d < D; d = d + 1) begin : g
      wire signed [7:0] q8 = q_flat[d*8 +: 8];
      wire signed [3:0] k4 = k_flat[d*4 +: 4];
      assign prods[d*32 +: 32] = q8 * k4;
    end
  endgenerate
  wire signed [31:0] acc;
  add_tree_bal #(.N(D), .W(32)) u_t (.xs(prods), .y(acc));
  assign score = acc >>> 8;
endmodule

module attn_sweep_pe #(
  parameter integer D     = 256,
  parameter integer P     = 4,
  parameter integer S_MAX = 16,
  parameter integer O_W   = 24
) (
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    en,
  input  wire [15:0]             seq_len,
  input  wire signed [D*8-1:0]   q_flat,
  input  wire                    wr_en,
  input  wire [15:0]             wr_t,
  input  wire [D*4-1:0]          wr_k,
  input  wire [D*4-1:0]          wr_v,
  output reg  signed [D*O_W-1:0] o_flat,
  output reg                     done,
  output wire                    ready
);
  localparam integer P_W    = (P <= 2) ? 1 : $clog2(P);
  localparam integer DEPTH  = S_MAX / P;
  localparam integer ADDR_W = (DEPTH <= 2) ? 1 : $clog2(DEPTH);
  localparam integer EXP_Q  = 8;
  localparam integer EXP_MAX = 32;
  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_ISS  = 2'd1;
  localparam [1:0] ST_EX   = 2'd2;

  wire [P_W-1:0]    wr_bank = wr_t[P_W-1:0];
  wire [ADDR_W-1:0] wr_addr = wr_t[15:P_W];

  wire [D*4-1:0] rdata_k [0:P-1];
  wire [D*4-1:0] rdata_v [0:P-1];
  reg  [ADDR_W-1:0] step_iss;
  reg  [ADDR_W-1:0] step_ex;
  reg               ex_v;
  reg  [1:0]        st;
  reg  [15:0]       S;
  reg  [ADDR_W-1:0] last_iss;
  reg  signed [D*8-1:0] q_r;
  reg               have;
  reg  signed [31:0] m_r;
  reg  signed [31:0] ell_r;
  reg  signed [31:0] o_r [0:D-1];

  genvar gp, gd;
  generate
    for (gp = 0; gp < P; gp = gp + 1) begin : g_bank
      wire we = wr_en && (wr_bank == gp[P_W-1:0]) && ready;
      kv_seq_ram #(.DEPTH(DEPTH), .WIDTH(D*4), .ADDR_W(ADDR_W)) u_k (
        .clk(clk), .we(we), .waddr(wr_addr), .wdata(wr_k),
        .raddr(step_iss), .rdata(rdata_k[gp])
      );
      kv_seq_ram #(.DEPTH(DEPTH), .WIDTH(D*4), .ADDR_W(ADDR_W)) u_v (
        .clk(clk), .we(we), .waddr(wr_addr), .wdata(wr_v),
        .raddr(step_iss), .rdata(rdata_v[gp])
      );
    end
  endgenerate

  wire signed [31:0] score [0:P-1];
  generate
    for (gp = 0; gp < P; gp = gp + 1) begin : g_dot
      attn_dot_i8i4 #(.D(D)) u_dot (
        .q_flat(q_r), .k_flat(rdata_k[gp]), .score(score[gp])
      );
    end
  endgenerate

  wire signed [31:0] m_blk;
  wire signed [P*32-1:0] score_flat;
  generate
    for (gp = 0; gp < P; gp = gp + 1) begin : g_sf
      assign score_flat[gp*32 +: 32] = score[gp];
    end
  endgenerate
  // P=4 max: two levels of compare (not a 512-ripple).
  wire signed [31:0] m01 = (score[0] > score[1]) ? score[0] : score[1];
  wire signed [31:0] m23 = (score[2] > score[3]) ? score[2] : score[3];
  assign m_blk = (m01 > m23) ? m01 : m23;
  wire signed [31:0] m_new = (!have) ? m_blk : ((m_blk > m_r) ? m_blk : m_r);

  wire [8:0] lut_s;
  wire [8:0] lut_w [0:P-1];
  wire signed [31:0] dlt = (m_new > m_r) ? (m_new - m_r) : 32'sd0;
  wire [5:0] dlt_u = (dlt > EXP_MAX) ? EXP_MAX[5:0] : dlt[5:0];
  attn_exp_lut u_sc (.d(dlt_u), .y(lut_s));
  generate
    for (gp = 0; gp < P; gp = gp + 1) begin : g_w
      wire signed [31:0] wd = m_new - score[gp];
      wire [5:0] wd_u = (wd < 0) ? 6'd0 : ((wd > EXP_MAX) ? EXP_MAX[5:0] : wd[5:0]);
      attn_exp_lut u_w (.d(wd_u), .y(lut_w[gp]));
    end
  endgenerate

  wire [8:0] scale = have ? lut_s : 9'd0;
  wire signed [31:0] wacc =
      $signed({23'd0, lut_w[0]}) + $signed({23'd0, lut_w[1]})
    + $signed({23'd0, lut_w[2]}) + $signed({23'd0, lut_w[3]});
  wire signed [31:0] ell_n = ((ell_r * $signed({23'd0, scale})) >>> EXP_Q) + wacc;

  wire signed [31:0] o_n [0:D-1];
  generate
    for (gd = 0; gd < D; gd = gd + 1) begin : g_o
      wire signed [3:0] v0 = rdata_v[0][gd*4 +: 4];
      wire signed [3:0] v1 = rdata_v[1][gd*4 +: 4];
      wire signed [3:0] v2 = rdata_v[2][gd*4 +: 4];
      wire signed [3:0] v3 = rdata_v[3][gd*4 +: 4];
      wire signed [31:0] acc0 = (o_r[gd] * $signed({23'd0, scale})) >>> EXP_Q;
      assign o_n[gd] = acc0
        + $signed({23'd0, lut_w[0]}) * {{28{v0[3]}}, v0}
        + $signed({23'd0, lut_w[1]}) * {{28{v1[3]}}, v1}
        + $signed({23'd0, lut_w[2]}) * {{28{v2[3]}}, v2}
        + $signed({23'd0, lut_w[3]}) * {{28{v3[3]}}, v3};
    end
  endgenerate

  function signed [31:0] tdiv;
    input signed [31:0] n;
    input signed [31:0] d;
    reg [31:0] ad;
    reg [31:0] an;
    begin
      ad = d[31] ? (~d + 1'b1) : d;
      if (ad == 32'd0) ad = 32'd1;
      an = n[31] ? (~n + 1'b1) : n;
      tdiv = n[31] ? -$signed(an / ad) : $signed(an / ad);
    end
  endfunction

  assign ready = (st == ST_IDLE);

  integer jj;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      step_iss <= {ADDR_W{1'b0}};
      step_ex  <= {ADDR_W{1'b0}};
      ex_v <= 1'b0;
      S <= 16'd1;
      last_iss <= {ADDR_W{1'b0}};
      o_flat <= {D*O_W{1'b0}};
      q_r <= {D*8{1'b0}};
      have <= 1'b0;
      m_r <= 32'sd0;
      ell_r <= 32'sd0;
      for (jj = 0; jj < D; jj = jj + 1) o_r[jj] <= 32'sd0;
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
              S <= s_cap[15:0];
              last_iss <= (n_steps <= 1) ? {ADDR_W{1'b0}} : (n_steps - 1);
            end
            step_iss <= {ADDR_W{1'b0}};
            q_r <= q_flat;
            have <= 1'b0;
            m_r <= 32'sd0;
            ell_r <= 32'sd0;
            for (jj = 0; jj < D; jj = jj + 1) o_r[jj] <= 32'sd0;
            st <= ST_ISS;
          end
        end
        ST_ISS: begin
          step_ex <= step_iss;
          ex_v <= 1'b1;
          if (step_iss == last_iss) st <= ST_EX;
          else step_iss <= step_iss + 1'b1;
        end
        ST_EX: begin
          ex_v <= 1'b0;
          for (jj = 0; jj < D; jj = jj + 1)
            o_flat[jj*O_W +: O_W] <= tdiv(o_n[jj] * 32'sd256, ell_n);
          done <= 1'b1;
          st <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
      if ((st == ST_ISS && ex_v) || (st == ST_EX)) begin
        have <= 1'b1;
        m_r <= m_new;
        ell_r <= ell_n;
        for (jj = 0; jj < D; jj = jj + 1) o_r[jj] <= o_n[jj];
      end
    end
  end
endmodule
