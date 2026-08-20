// 0.8B mixer farm: one BRAM PE, 24 layers in time, independent S per layer.
// 16 K-heads × 3 V-heads overlap (gated_delta_gqa3): layer cycles = one PE.
// Serial V-heads would be 3×; that was the old "heads do not cut cycles" bug.
// Qwen3.8-27B fused mixer is D=128 at D+2 = 130 clk (rtl/gated_delta_fused.v).
// FFN/attn are not in this stub; model tok/s = Fmax / (N_LAYERS * pe_cycles).
`timescale 1ns / 1ps

module qwen08b_delta_farm #(
  parameter integer D        = 4,
  parameter integer N_LAYERS = 24,
  parameter integer QK_W     = 8,
  parameter integer V_W      = 8,
  parameter integer G_W      = 8,
  parameter integer O_W      = 24
) (
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     en,
  input  wire signed [D*QK_W-1:0] q_flat,
  input  wire signed [D*QK_W-1:0] k_flat,
  input  wire signed [D*V_W-1:0]  v_flat,
  input  wire        [G_W-1:0]    g,
  input  wire        [G_W-1:0]    beta,
  output wire signed [D*O_W-1:0]  o_flat,
  output reg                      done,
  output wire                     ready
);
  localparam integer LW = (N_LAYERS <= 1) ? 1 : $clog2(N_LAYERS);
  localparam [LW-1:0] LAST_L = N_LAYERS - 1;

  localparam ST_IDLE = 1'b0;
  localparam ST_RUN  = 1'b1;

  reg st;
  reg pe_en;
  reg pending;
  reg [LW-1:0] layer;
  wire pe_done;
  wire pe_ready;

  assign ready = (st == ST_IDLE) && pe_ready;

  gated_delta_bram #(.D(D), .N_LAYERS(N_LAYERS)) u_pe (
    .clk(clk),
    .rst_n(rst_n),
    .en(pe_en),
    .layer(layer),
    .q_flat(q_flat),
    .k_flat(k_flat),
    .v_flat(v_flat),
    .g(g),
    .beta(beta),
    .o_flat(o_flat),
    .done(pe_done),
    .ready(pe_ready)
  );

  always @(posedge clk) begin
    done  <= 1'b0;
    pe_en <= 1'b0;
    if (!rst_n) begin
      st      <= ST_IDLE;
      layer   <= {LW{1'b0}};
      pending <= 1'b0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en && pe_ready) begin
            layer   <= {LW{1'b0}};
            pe_en   <= 1'b1;
            pending <= 1'b0;
            st      <= ST_RUN;
          end
        end
        ST_RUN: begin
          if (pe_done) begin
            if (layer == LAST_L) begin
              done    <= 1'b1;
              pending <= 1'b0;
              st      <= ST_IDLE;
            end else begin
              layer   <= layer + 1'b1;
              pending <= 1'b1;
            end
          end
          if (pending && pe_ready) begin
            pe_en   <= 1'b1;
            pending <= 1'b0;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
