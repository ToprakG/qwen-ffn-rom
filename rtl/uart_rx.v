// 8N1 UART RX. Sample mid-bit. valid is 1 cycle.
`timescale 1ns / 1ps

module uart_rx #(
  parameter integer DIV = 33
) (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       rx,
  output reg  [7:0] data,
  output reg        valid
);
  localparam integer DW = (DIV <= 2) ? 1 : $clog2(DIV);

  reg [1:0] rx_s;
  reg [2:0] st;
  reg [DW-1:0] div;
  reg [2:0] bit_i;
  reg [7:0] sh;

  localparam ST_IDLE  = 3'd0;
  localparam ST_START = 3'd1;
  localparam ST_DATA  = 3'd2;
  localparam ST_STOP  = 3'd3;

  always @(posedge clk) begin
    valid <= 1'b0;
    rx_s  <= {rx_s[0], rx};
    if (!rst_n) begin
      st   <= ST_IDLE;
      div  <= {DW{1'b0}};
      bit_i <= 3'd0;
      data <= 8'd0;
      sh   <= 8'd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (rx_s == 2'b10) begin
            div <= DIV / 2;
            st  <= ST_START;
          end
        end
        ST_START: begin
          if (div == 0) begin
            div   <= DIV - 1;
            bit_i <= 3'd0;
            st    <= rx_s[0] ? ST_IDLE : ST_DATA;
          end else
            div <= div - 1'b1;
        end
        ST_DATA: begin
          if (div == 0) begin
            sh    <= {rx_s[0], sh[7:1]};
            div   <= DIV - 1;
            if (bit_i == 3'd7)
              st <= ST_STOP;
            else
              bit_i <= bit_i + 1'b1;
          end else
            div <= div - 1'b1;
        end
        ST_STOP: begin
          if (div == 0) begin
            data  <= sh;
            valid <= 1'b1;
            st    <= ST_IDLE;
          end else
            div <= div - 1'b1;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
