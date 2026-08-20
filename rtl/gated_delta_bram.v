// FPGA Gated DeltaNet PE: D column BRAMs, one DSP48 per lane (25×18).
// Operands: S×g (16×9), S×k / S×q (16×8), k×delta and β×(v−kv) as 25×8.
// |delta| and |v−kv| fit in 25 bits because S is saturating 16-bit.
// BRAM read is 1 cycle; issue the next row while multiplying the current one
// so a sweep is D+2 cycles (~4D+9 per layer), not 3D.
// Same integer math as quant/delta_int.py.
`timescale 1ns / 1ps

module gated_delta_bram #(
  parameter integer D         = 16,
  parameter integer N_LAYERS  = 1,
  parameter integer QK_W      = 8,
  parameter integer V_W       = 8,
  parameter integer S_W       = 16,
  parameter integer G_W       = 8,
  parameter integer SHIFT     = 8,
  parameter integer O_W       = 24
) (
  input  wire                         clk,
  input  wire                         rst_n,
  input  wire                         en,
  input  wire [(N_LAYERS <= 1 ? 1 : $clog2(N_LAYERS))-1:0] layer,
  input  wire signed [D*QK_W-1:0]     q_flat,
  input  wire signed [D*QK_W-1:0]     k_flat,
  input  wire signed [D*V_W-1:0]      v_flat,
  input  wire        [G_W-1:0]        g,
  input  wire        [G_W-1:0]        beta,
  output reg  signed [D*O_W-1:0]      o_flat,
  output reg                          done,
  output wire                         ready
);
  localparam integer DEPTH  = N_LAYERS * D;
  localparam integer AW     = $clog2(DEPTH);
  localparam integer LW     = (N_LAYERS <= 1) ? 1 : $clog2(N_LAYERS);
  localparam integer IDX_W  = (D <= 2) ? 1 : $clog2(D);
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
  localparam [2:0] ST_CLEAR = 3'd7;

  localparam [1:0] PIPE_FILL  = 2'd0;
  localparam [1:0] PIPE_RUN   = 2'd1;
  localparam [1:0] PIPE_DRAIN = 2'd2;

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
  reg [LW-1:0]          layer_r;

  reg signed [31:0] kv [0:D-1];
  reg signed [31:0] delta [0:D-1];
  reg signed [31:0] acc [0:D-1];
  reg signed [O_W-1:0] o_r [0:D-1];
  reg signed [S_W-1:0] s_hold [0:D-1];
  (* use_dsp = "yes", use_dsp48 = "yes" *)
  reg signed [31:0] prod [0:D-1];

  reg [2:0] st;
  reg [1:0] pipe;
  reg [IDX_W-1:0] row_iss;
  reg [IDX_W-1:0] row_mul;
  reg [IDX_W-1:0] row_wb;
  reg mul_v;
  reg wb_v;
  reg [AW:0] clr_i;

  wire [AW-1:0] rd_addr = layer_r * D + row_iss;

  reg                      we;
  reg  [AW-1:0]            waddr_r;
  reg  signed [S_W-1:0]    wdata_c [0:D-1];
  wire signed [S_W-1:0]    rdata [0:D-1];
  wire signed [S_W-1:0]    s_el [0:D-1];
  wire signed [S_W-1:0]    s_sat [0:D-1];
  wire signed [S_W-1:0]    decay_w [0:D-1];
  wire signed [31:0]       v_sx [0:D-1];

  genvar t;
  generate
    for (t = 0; t < D; t = t + 1) begin : lanes
      delta_s_col_ram #(
        .DEPTH(DEPTH),
        .S_W(S_W),
        .ADDR_W(AW)
      ) u_s (
        .clk(clk),
        .we(we),
        .waddr(waddr_r),
        .wdata(wdata_c[t]),
        .raddr(rd_addr),
        .rdata(rdata[t])
      );

      assign s_el[t]  = rdata[t];
      assign v_sx[t]  = v_r[t];

      // One 25×18 multiply per lane. Mux operands; never instantiate a second *.
      reg  signed [24:0] mula;
      reg  signed [17:0] mulb;
      wire signed [42:0] mul_p = mula * mulb;

      always @* begin
        mula = 25'sd0;
        mulb = 18'sd0;
        case (st)
          ST_DECAY: begin
            mula = s_el[t];
            mulb = g_r;
          end
          ST_KV: begin
            mula = s_el[t];
            mulb = k_r[row_mul];
          end
          ST_DELTA: begin
            mula = v_sx[t] - kv[t];
            mulb = beta_r;
          end
          ST_OUTER: begin
            mula = delta[t];
            mulb = k_r[row_mul];
          end
          ST_OUT: begin
            mula = s_el[t];
            mulb = q_r[row_mul];
          end
          default: begin
            mula = 25'sd0;
            mulb = 18'sd0;
          end
        endcase
      end

      always @(posedge clk) begin
        if (!rst_n)
          prod[t] <= 32'sd0;
        else
          prod[t] <= mul_p;
      end

      wire signed [31:0] s_h     = s_hold[t];
      wire signed [31:0] outer_s = s_h + (prod[t] >>> SHIFT);
      assign s_sat[t]   = (outer_s > S_MAX) ? S_MAX[S_W-1:0] :
                          (outer_s < S_MIN) ? S_MIN[S_W-1:0] :
                          outer_s[S_W-1:0];
      assign decay_w[t] = prod[t] >>> SHIFT;
    end
  endgenerate

  assign ready = (st == ST_IDLE);

  integer ii;
  always @(posedge clk) begin
    done <= 1'b0;
    we   <= 1'b0;
    if (!rst_n) begin
      st      <= ST_CLEAR;
      pipe    <= PIPE_FILL;
      row_iss <= {IDX_W{1'b0}};
      row_mul <= {IDX_W{1'b0}};
      row_wb  <= {IDX_W{1'b0}};
      mul_v   <= 1'b0;
      wb_v    <= 1'b0;
      clr_i   <= {(AW+1){1'b0}};
      waddr_r <= {AW{1'b0}};
      o_flat  <= {D * O_W{1'b0}};
      g_r     <= {(G_W+1){1'b0}};
      beta_r  <= {(G_W+1){1'b0}};
      layer_r <= {LW{1'b0}};
      for (ii = 0; ii < D; ii = ii + 1) begin
        kv[ii]      <= 32'sd0;
        delta[ii]   <= 32'sd0;
        acc[ii]     <= 32'sd0;
        o_r[ii]     <= {O_W{1'b0}};
        q_r[ii]     <= {QK_W{1'b0}};
        k_r[ii]     <= {QK_W{1'b0}};
        v_r[ii]     <= {V_W{1'b0}};
        s_hold[ii]  <= {S_W{1'b0}};
        wdata_c[ii] <= {S_W{1'b0}};
      end
    end else begin
      case (st)
        ST_CLEAR: begin
          if (clr_i < DEPTH) begin
            we      <= 1'b1;
            waddr_r <= clr_i[AW-1:0];
            for (ii = 0; ii < D; ii = ii + 1)
              wdata_c[ii] <= {S_W{1'b0}};
            clr_i <= clr_i + 1'b1;
          end else begin
            st    <= ST_IDLE;
            clr_i <= {(AW+1){1'b0}};
          end
        end

        ST_IDLE: begin
          pipe    <= PIPE_FILL;
          row_iss <= {IDX_W{1'b0}};
          mul_v   <= 1'b0;
          wb_v    <= 1'b0;
          if (en) begin
            for (ii = 0; ii < D; ii = ii + 1) begin
              q_r[ii] <= q_i[ii];
              k_r[ii] <= k_i[ii];
              v_r[ii] <= v_i[ii];
            end
            g_r     <= {1'b0, g};
            beta_r  <= {1'b0, beta};
            layer_r <= layer;
            st      <= ST_DECAY;
          end
        end

        ST_DECAY, ST_KV, ST_OUTER, ST_OUT: begin
          if (mul_v) begin
            for (ii = 0; ii < D; ii = ii + 1)
              s_hold[ii] <= s_el[ii];
          end

          if (wb_v) begin
            if (st == ST_DECAY) begin
              we      <= 1'b1;
              waddr_r <= layer_r * D + row_wb;
              for (ii = 0; ii < D; ii = ii + 1)
                wdata_c[ii] <= decay_w[ii];
            end else if (st == ST_KV) begin
              if (row_wb == {IDX_W{1'b0}}) begin
                for (ii = 0; ii < D; ii = ii + 1)
                  acc[ii] <= prod[ii];
              end else if (row_wb == LAST) begin
                for (ii = 0; ii < D; ii = ii + 1)
                  kv[ii] <= (acc[ii] + prod[ii]) >>> SHIFT;
              end else begin
                for (ii = 0; ii < D; ii = ii + 1)
                  acc[ii] <= acc[ii] + prod[ii];
              end
            end else if (st == ST_OUTER) begin
              we      <= 1'b1;
              waddr_r <= layer_r * D + row_wb;
              for (ii = 0; ii < D; ii = ii + 1)
                wdata_c[ii] <= s_sat[ii];
            end else begin
              if (row_wb == {IDX_W{1'b0}}) begin
                for (ii = 0; ii < D; ii = ii + 1)
                  acc[ii] <= prod[ii];
              end else if (row_wb == LAST) begin
                for (ii = 0; ii < D; ii = ii + 1)
                  o_r[ii] <= (acc[ii] + prod[ii]) >>> SHIFT;
              end else begin
                for (ii = 0; ii < D; ii = ii + 1)
                  acc[ii] <= acc[ii] + prod[ii];
              end
            end
          end

          case (pipe)
            PIPE_FILL: begin
              row_mul <= row_iss;
              mul_v   <= 1'b1;
              wb_v    <= 1'b0;
              if (row_iss != LAST)
                row_iss <= row_iss + 1'b1;
              pipe <= PIPE_RUN;
            end
            PIPE_RUN: begin
              row_wb  <= row_mul;
              wb_v    <= 1'b1;
              row_mul <= row_iss;
              if (row_mul == LAST) begin
                mul_v <= 1'b0;
                pipe  <= PIPE_DRAIN;
              end else if (row_iss != LAST)
                row_iss <= row_iss + 1'b1;
            end
            PIPE_DRAIN: begin
              wb_v    <= 1'b0;
              mul_v   <= 1'b0;
              row_iss <= {IDX_W{1'b0}};
              pipe    <= PIPE_FILL;
              if (st == ST_DECAY)
                st <= ST_KV;
              else if (st == ST_KV)
                st <= ST_DELTA;
              else if (st == ST_OUTER)
                st <= ST_OUT;
              else
                st <= ST_PACK;
            end
            default: pipe <= PIPE_FILL;
          endcase
        end

        ST_DELTA: begin
          // Same DSP as the row sweeps: wait one cycle for mula*mulb to land in prod.
          if (!wb_v) begin
            wb_v <= 1'b1;
          end else begin
            for (ii = 0; ii < D; ii = ii + 1)
              delta[ii] <= prod[ii] >>> SHIFT;
            row_iss <= {IDX_W{1'b0}};
            pipe    <= PIPE_FILL;
            mul_v   <= 1'b0;
            wb_v    <= 1'b0;
            st      <= ST_OUTER;
          end
        end

        ST_PACK: begin
          for (ii = 0; ii < D; ii = ii + 1)
            o_flat[ii*O_W +: O_W] <= o_r[ii];
          done <= 1'b1;
          st   <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
