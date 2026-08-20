// Gated DeltaNet PE — column-stream, fused update + readout.
//
// S is (D, D): axis 0 = key, axis 1 = value. Banked as D row SRAMs addressed
// by column, so one cycle reads S[:, j] (D lanes). The 524-clk BRAM PE already
// had D-wide rows; 524 was four sequential sweeps (decay, kv, outer, q) plus
// handshake. This PE issues one column per cycle:
//
//   s_dec = (S[:,j] * g) >> 8
//   kv    = (s_dec · k) >> 8
//   delta = (β * (v[j] − kv)) >> 8
//   S'    = sat(s_dec + (k * delta) >> 8)
//   o[j]  = (S' · q) >> 8
//
// SRAM read is 1 cycle: issue col j, next cycle compute+write. 1-issue/cycle,
// no per-step handshake. Cycles = D + 2 (issue fill + last write).
// Bandwidth floor: D² elements / D lanes / 1 pass = D. P < D cannot hit ~130
// at D=128 on 1R1W SRAM — that is the floor, not a scheduling bug.
//
// Bit-exact vs quant/delta_int.py gated_delta_step.
`timescale 1ns / 1ps

// Balanced adder tree (heap layout). A sequential acc loop synthesizes as a
// 128-long ripple at D=128 (~1.5 µs ABC). Associative wrapping 32-bit add.
module gated_delta_fused_dot #(
  parameter integer N = 16,
  parameter integer W = 32
) (
  input  wire signed [N*W-1:0] xs,
  output wire signed [W-1:0]   y
);
  add_tree_bal #(.N(N), .W(W)) u_tree (.xs(xs), .y(y));
endmodule

module gated_delta_fused #(
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
  output reg                      done,
  output wire                     ready
);
  localparam integer IDX_W = (D <= 2) ? 1 : $clog2(D);
  localparam [IDX_W-1:0] LAST = D - 1;
  localparam signed [31:0] S_MAX = 32'sd32767;
  localparam signed [31:0] S_MIN = -32'sd32768;

  localparam [1:0] ST_IDLE = 2'd0;
  localparam [1:0] ST_ISS  = 2'd1;
  localparam [1:0] ST_EX   = 2'd2;

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

  reg [1:0] st;
  reg [IDX_W-1:0] col_iss;
  reg [IDX_W-1:0] col_ex;
  reg ex_v;

  wire signed [S_W-1:0] rdata [0:D-1];
  reg  signed [S_W-1:0] wdata [0:D-1];
  reg                   we;
  reg  [IDX_W-1:0]      wcol;

  genvar t;
  generate
    for (t = 0; t < D; t = t + 1) begin : banks
      delta_s_col_ram #(
        .DEPTH(D),
        .S_W(S_W),
        .ADDR_W(IDX_W)
      ) u_s (
        .clk(clk),
        .we(we),
        .waddr(wcol),
        .wdata(wdata[t]),
        .raddr(col_iss),
        .rdata(rdata[t])
      );
    end
  endgenerate

  wire signed [S_W-1:0] s_dec [0:D-1];
  wire signed [S_W-1:0] s_nxt [0:D-1];
  wire signed [D*32-1:0] kv_flat;
  wire signed [D*32-1:0] o_prod_flat;
  wire signed [31:0] kv_sum;
  wire signed [31:0] o_sum;
  wire signed [31:0] kv_s;
  wire signed [31:0] delta_s;

  generate
    for (t = 0; t < D; t = t + 1) begin : lanes
      wire signed [31:0] s_ext = {{(32-S_W){rdata[t][S_W-1]}}, rdata[t]};
      wire signed [31:0] g_ext = {{(32-(G_W+1)){g_r[G_W]}}, g_r};
      wire signed [31:0] k_ext = {{(32-QK_W){k_r[t][QK_W-1]}}, k_r[t]};
      wire signed [31:0] dec32 = (s_ext * g_ext) >>> SHIFT;
      assign s_dec[t] = dec32[S_W-1:0];
      wire signed [31:0] dec_ext = {{(32-S_W){s_dec[t][S_W-1]}}, s_dec[t]};
      assign kv_flat[t*32 +: 32] = dec_ext * k_ext;
    end
  endgenerate

  gated_delta_fused_dot #(.N(D), .W(32)) u_kv (
    .xs(kv_flat),
    .y(kv_sum)
  );

  assign kv_s = kv_sum >>> SHIFT;
  wire signed [31:0] v_ext_c = {{(32-V_W){v_r[col_ex][V_W-1]}}, v_r[col_ex]};
  wire signed [31:0] b_ext   = {{(32-(G_W+1)){beta_r[G_W]}}, beta_r};
  assign delta_s = (b_ext * (v_ext_c - kv_s)) >>> SHIFT;

  generate
    for (t = 0; t < D; t = t + 1) begin : outer
      wire signed [31:0] k_ext = {{(32-QK_W){k_r[t][QK_W-1]}}, k_r[t]};
      wire signed [31:0] q_ext = {{(32-QK_W){q_r[t][QK_W-1]}}, q_r[t]};
      wire signed [31:0] dec_ext = {{(32-S_W){s_dec[t][S_W-1]}}, s_dec[t]};
      wire signed [31:0] outer_s = dec_ext + ((k_ext * delta_s) >>> SHIFT);
      assign s_nxt[t] = (outer_s > S_MAX) ? S_MAX[S_W-1:0] :
                        (outer_s < S_MIN) ? S_MIN[S_W-1:0] :
                        outer_s[S_W-1:0];
      wire signed [31:0] nxt_ext = {{(32-S_W){s_nxt[t][S_W-1]}}, s_nxt[t]};
      assign o_prod_flat[t*32 +: 32] = nxt_ext * q_ext;
    end
  endgenerate

  gated_delta_fused_dot #(.N(D), .W(32)) u_o (
    .xs(o_prod_flat),
    .y(o_sum)
  );

  wire signed [O_W-1:0] o_col = o_sum >>> SHIFT;

  assign ready = (st == ST_IDLE);

  integer ii;
  always @(posedge clk) begin
    done <= 1'b0;
    we   <= 1'b0;
    if (!rst_n) begin
      st       <= ST_IDLE;
      col_iss  <= {IDX_W{1'b0}};
      col_ex   <= {IDX_W{1'b0}};
      ex_v     <= 1'b0;
      o_flat   <= {D * O_W{1'b0}};
      g_r      <= {(G_W+1){1'b0}};
      beta_r   <= {(G_W+1){1'b0}};
      wcol     <= {IDX_W{1'b0}};
      for (ii = 0; ii < D; ii = ii + 1) begin
        q_r[ii]   <= {QK_W{1'b0}};
        k_r[ii]   <= {QK_W{1'b0}};
        v_r[ii]   <= {V_W{1'b0}};
        wdata[ii] <= {S_W{1'b0}};
      end
    end else begin
      case (st)
        ST_IDLE: begin
          ex_v <= 1'b0;
          if (en) begin
            for (ii = 0; ii < D; ii = ii + 1) begin
              q_r[ii] <= q_i[ii];
              k_r[ii] <= k_i[ii];
              v_r[ii] <= v_i[ii];
            end
            g_r     <= {1'b0, g};
            beta_r  <= {1'b0, beta};
            col_iss <= {IDX_W{1'b0}};
            st      <= ST_ISS;
          end
        end

        ST_ISS: begin
          // col_iss is on raddr this cycle; rdata lands next cycle.
          col_ex  <= col_iss;
          ex_v    <= 1'b1;
          if (col_iss == LAST)
            st <= ST_EX;
          else
            col_iss <= col_iss + 1'b1;
        end

        ST_EX: begin
          // Last column: rdata valid, write, pack o, done.
          ex_v <= 1'b0;
          we   <= 1'b1;
          wcol <= col_ex;
          for (ii = 0; ii < D; ii = ii + 1)
            wdata[ii] <= s_nxt[ii];
          o_flat[col_ex*O_W +: O_W] <= o_col;
          done <= 1'b1;
          st   <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase

      // Steady-state: every ISS cycle after the first also writes the
      // previous column (rdata of col_iss-1 / col_ex).
      if (st == ST_ISS && ex_v) begin
        we   <= 1'b1;
        wcol <= col_ex;
        for (ii = 0; ii < D; ii = ii + 1)
          wdata[ii] <= s_nxt[ii];
        o_flat[col_ex*O_W +: O_W] <= o_col;
      end
    end
  end
endmodule
