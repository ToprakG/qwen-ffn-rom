// Handshake between the 27B layer blocks. Datapaths are outside: this FSM
// only sees done/ready. Cycle counts live in the PEs (mixer D+2, RMS 2,
// attn ceil(S/P)+2, FFN farm-hidden).
`timescale 1ns / 1ps

module layer_sequencer (
  input  wire clk,
  input  wire rst_n,
  input  wire en,
  input  wire mix_sel,
  input  wire rms_done,
  input  wire body_done,
  input  wire body_ready,
  input  wire ffn_done,
  output reg  rms_en,
  output reg  body_en,
  output reg  ffn_en,
  output reg  done,
  output wire ready
);
  localparam [3:0] ST_IDLE = 4'd0;
  localparam [3:0] ST_N1E  = 4'd1;
  localparam [3:0] ST_N1   = 4'd2;
  localparam [3:0] ST_BE   = 4'd3;
  localparam [3:0] ST_BODY = 4'd4;
  localparam [3:0] ST_N2E  = 4'd5;
  localparam [3:0] ST_N2   = 4'd6;
  localparam [3:0] ST_FE   = 4'd7;
  localparam [3:0] ST_FFN  = 4'd8;

  reg [3:0] st;
  assign ready = (st == ST_IDLE);

  always @(posedge clk) begin
    rms_en  <= 1'b0;
    body_en <= 1'b0;
    ffn_en  <= 1'b0;
    done    <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) st <= ST_N1E;
        end
        ST_N1E: begin
          rms_en <= 1'b1;
          st     <= ST_N1;
        end
        ST_N1: begin
          if (rms_done) st <= ST_BE;
        end
        ST_BE: begin
          if (!mix_sel || body_ready) begin
            body_en <= 1'b1;
            st      <= ST_BODY;
          end
        end
        ST_BODY: begin
          if (body_done) st <= ST_N2E;
        end
        ST_N2E: begin
          rms_en <= 1'b1;
          st     <= ST_N2;
        end
        ST_N2: begin
          if (rms_done) st <= ST_FE;
        end
        ST_FE: begin
          ffn_en <= 1'b1;
          st     <= ST_FFN;
        end
        ST_FFN: begin
          if (ffn_done) begin
            done <= 1'b1;
            st   <= ST_IDLE;
          end
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
