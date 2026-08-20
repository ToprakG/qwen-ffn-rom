// One complete Qwen decoder layer (silicon-ready slice).
// H=16 = 4 heads × D=4. FFN is the 0.8B down_proj 16×16 CSD tile.
// rms1 → (DeltaNet | gated-attn MAC) per head → residual → rms2 → FFN → residual.
// Mixer is the signed-off flop PE (gated_delta_step), not FPGA BRAM.
// Bit-exact vs quant/layer_int.py qwen_layer_int.
`timescale 1ns / 1ps

module qwen_layer #(
  parameter integer H     = 16,
  parameter integer D     = 4,
  parameter integer HEADS = 4,
  parameter integer S_MAX = 8,
  parameter integer O_W   = 24,
  parameter integer SHIFT = 8,
  parameter integer ACC_W = 16
) (
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  en,
  input  wire                  use_attn,
  input  wire signed [H*8-1:0] x_flat,
  input  wire        [7:0]     g,
  input  wire        [7:0]     beta,
  input  wire signed [H*8-1:0] w_n1_flat,
  input  wire signed [H*8-1:0] w_n2_flat,
  output reg  signed [H*8-1:0] y_flat,
  output reg                   done,
  output wire                  ready
);
  localparam [3:0] ST_IDLE  = 4'd0;
  localparam [3:0] ST_N1E   = 4'd1;
  localparam [3:0] ST_N1    = 4'd2;
  localparam [3:0] ST_HEADE = 4'd3;
  localparam [3:0] ST_ATTNK = 4'd11;
  localparam [3:0] ST_HEAD  = 4'd4;
  localparam [3:0] ST_HEADW = 4'd12;
  localparam [3:0] ST_RES1  = 4'd5;
  localparam [3:0] ST_N2E   = 4'd6;
  localparam [3:0] ST_N2    = 4'd7;
  localparam [3:0] ST_FFNE  = 4'd8;
  localparam [3:0] ST_FFN   = 4'd9;
  localparam [3:0] ST_OUT   = 4'd10;

  integer k;
  integer hh;
  integer tt;
  integer dd;
  reg [3:0] st;
  reg use_attn_r;
  reg [7:0] g_r;
  reg [7:0] beta_r;
  reg signed [7:0] x_r [0:H-1];
  reg signed [7:0] mid [0:H-1];
  reg [1:0] head;
  reg [7:0] s_len [0:HEADS-1];
  reg signed [7:0] k_mem [0:HEADS-1][0:S_MAX-1][0:D-1];
  reg signed [7:0] v_mem [0:HEADS-1][0:S_MAX-1][0:D-1];

  wire signed [H*8-1:0] x_now;
  wire signed [H*8-1:0] mid_now;
  genvar gi, gj;
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : pk
      assign x_now[gi*8 +: 8] = x_r[gi];
      assign mid_now[gi*8 +: 8] = mid[gi];
    end
  endgenerate

  reg n1_en, n2_en;
  wire n1_done, n2_done;
  wire signed [H*8-1:0] h1, h2;

  rmsnorm #(.H(H)) u_n1 (
    .clk(clk), .rst_n(rst_n), .en(n1_en),
    .x_flat(x_now), .w_flat(w_n1_flat), .y_flat(h1), .done(n1_done)
  );
  rmsnorm #(.H(H)) u_n2 (
    .clk(clk), .rst_n(rst_n), .en(n2_en),
    .x_flat(mid_now), .w_flat(w_n2_flat), .y_flat(h2), .done(n2_done)
  );

  reg signed [H*8-1:0] h1_cap;

  wire signed [7:0] h1_i [0:H-1];
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : uh
      assign h1_i[gi] = h1[gi*8 +: 8];
    end
  endgenerate

  reg mix_en [0:HEADS-1];
  wire mix_done [0:HEADS-1];
  wire signed [D*O_W-1:0] mix_o [0:HEADS-1];
  generate
    for (gi = 0; gi < HEADS; gi = gi + 1) begin : mix
      gated_delta_step #(.D(D), .O_W(O_W), .SHIFT(SHIFT)) u_mix (
        .clk(clk), .rst_n(rst_n), .en(mix_en[gi]),
        .q_flat(h1[gi*D*8 +: D*8]),
        .k_flat(h1[gi*D*8 +: D*8]),
        .v_flat(h1[gi*D*8 +: D*8]),
        .g(g_r), .beta(beta_r),
        .o_flat(mix_o[gi]), .done(mix_done[gi])
      );
    end
  endgenerate

  wire signed [S_MAX*D*8-1:0] attn_k_flat;
  wire signed [S_MAX*D*8-1:0] attn_v_flat;
  generate
    for (gi = 0; gi < S_MAX; gi = gi + 1) begin : pkv
      for (gj = 0; gj < D; gj = gj + 1) begin : pkd
        assign attn_k_flat[(gi*D+gj)*8 +: 8] = k_mem[head][gi][gj];
        assign attn_v_flat[(gi*D+gj)*8 +: 8] = v_mem[head][gi][gj];
      end
    end
  endgenerate

  reg attn_en;
  wire attn_done;
  wire signed [D*O_W-1:0] attn_o;
  attn_decode #(.D(D), .S_MAX(S_MAX), .O_W(O_W), .SHIFT(SHIFT)) u_attn (
    .clk(clk), .rst_n(rst_n), .en(attn_en),
    .seq_len(s_len[head]),
    .q_flat(h1[head*D*8 +: D*8]),
    .k_flat(attn_k_flat), .v_flat(attn_v_flat),
    .o_flat(attn_o), .done(attn_done)
  );

  wire signed [H*8-1:0] h2_now = h2;
  wire signed [H*ACC_W-1:0] y_c;
  ffn_tile u_ffn (
    .x_flat(h2_now),
    .y_flat(y_c)
  );

  wire signed [O_W-1:0] mix_lane0 [0:D-1];
  wire signed [O_W-1:0] mix_lane1 [0:D-1];
  wire signed [O_W-1:0] mix_lane2 [0:D-1];
  wire signed [O_W-1:0] mix_lane3 [0:D-1];
  wire signed [O_W-1:0] attn_lane [0:D-1];
  generate
    for (gi = 0; gi < D; gi = gi + 1) begin : ln
      assign mix_lane0[gi] = mix_o[0][gi*O_W +: O_W];
      assign mix_lane1[gi] = mix_o[1][gi*O_W +: O_W];
      assign mix_lane2[gi] = mix_o[2][gi*O_W +: O_W];
      assign mix_lane3[gi] = mix_o[3][gi*O_W +: O_W];
      assign attn_lane[gi] = attn_o[gi*O_W +: O_W];
    end
  endgenerate

  wire signed [O_W-1:0] mix_pick [0:D-1];
  generate
    for (gi = 0; gi < D; gi = gi + 1) begin : pkhd
      assign mix_pick[gi] = (head == 2'd0) ? mix_lane0[gi] :
                            (head == 2'd1) ? mix_lane1[gi] :
                            (head == 2'd2) ? mix_lane2[gi] :
                            mix_lane3[gi];
    end
  endgenerate

  wire signed [31:0] res_m [0:D-1];
  wire signed [7:0]  sat_m [0:D-1];
  generate
    for (gi = 0; gi < D; gi = gi + 1) begin : rm
      wire signed [O_W-1:0] src = use_attn_r ? attn_lane[gi] : mix_pick[gi];
      assign res_m[gi] = x_r[head*D+gi] + (src >>> SHIFT);
      assign sat_m[gi] = (res_m[gi] > 32'sd127) ? 8'sd127 :
                         (res_m[gi] < -32'sd128) ? -8'sd128 : res_m[gi][7:0];
    end
  endgenerate

  wire signed [ACC_W-1:0] y_lane [0:H-1];
  wire signed [31:0] res2 [0:H-1];
  wire signed [7:0]  ysat [0:H-1];
  generate
    for (gi = 0; gi < H; gi = gi + 1) begin : r2
      assign y_lane[gi] = y_c[gi*ACC_W +: ACC_W];
      assign res2[gi] = mid[gi] + (y_lane[gi] >>> 7);
      assign ysat[gi] = (res2[gi] > 32'sd127) ? 8'sd127 :
                        (res2[gi] < -32'sd128) ? -8'sd128 : res2[gi][7:0];
    end
  endgenerate

  assign ready = (st == ST_IDLE);

  always @(posedge clk) begin
    done   <= 1'b0;
    n1_en  <= 1'b0;
    n2_en  <= 1'b0;
    attn_en <= 1'b0;
    for (hh = 0; hh < HEADS; hh = hh + 1)
      mix_en[hh] <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      use_attn_r <= 1'b0;
      g_r <= 8'd0;
      beta_r <= 8'd0;
      head <= 2'd0;
      y_flat <= {H*8{1'b0}};
      for (k = 0; k < H; k = k + 1) begin
        x_r[k] <= 8'sd0;
        mid[k] <= 8'sd0;
      end
      for (hh = 0; hh < HEADS; hh = hh + 1) begin
        s_len[hh] <= 8'd0;
        for (tt = 0; tt < S_MAX; tt = tt + 1)
          for (dd = 0; dd < D; dd = dd + 1) begin
            k_mem[hh][tt][dd] <= 8'sd0;
            v_mem[hh][tt][dd] <= 8'sd0;
          end
      end
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (k = 0; k < H; k = k + 1)
              x_r[k] <= x_flat[k*8 +: 8];
            g_r        <= g;
            beta_r     <= beta;
            use_attn_r <= use_attn;
            head       <= 2'd0;
            st         <= ST_N1E;
          end
        end
        ST_N1E: begin
          n1_en <= 1'b1;
          st    <= ST_N1;
        end
        ST_N1: begin
          if (n1_done)
            st <= ST_HEADE;
        end
        ST_HEADE: begin
          if (head == 2'd0)
            h1_cap <= h1;
          if (use_attn_r) begin
            if (s_len[head] < S_MAX) begin
              for (dd = 0; dd < D; dd = dd + 1) begin
                k_mem[head][s_len[head]][dd] <= h1_i[head*D+dd];
                v_mem[head][s_len[head]][dd] <= h1_i[head*D+dd];
              end
              s_len[head] <= s_len[head] + 8'd1;
            end
            st <= ST_ATTNK;
          end else begin
            mix_en[head] <= 1'b1;
            st <= ST_HEAD;
          end
        end
        ST_ATTNK: begin
          attn_en <= 1'b1;
          st      <= ST_HEAD;
        end
        ST_HEAD: begin
          if (use_attn_r) begin
            if (attn_done)
              st <= ST_HEADW;
          end else if (mix_done[head])
            st <= ST_HEADW;
        end
        ST_HEADW: begin
          st <= ST_RES1;
        end
        ST_RES1: begin
          for (dd = 0; dd < D; dd = dd + 1)
            mid[head*D+dd] <= sat_m[dd];
          if (head == HEADS - 1)
            st <= ST_N2E;
          else begin
            head <= head + 2'd1;
            st   <= ST_HEADE;
          end
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
          st <= ST_FFN;
        end
        ST_FFN: begin
          st <= ST_OUT;
        end
        ST_OUT: begin
          for (k = 0; k < H; k = k + 1)
            y_flat[k*8 +: 8] <= ysat[k];
          done <= 1'b1;
          st   <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
