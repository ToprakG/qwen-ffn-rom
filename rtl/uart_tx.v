// 8N1 UART TX. start is 1 cycle while !busy.
`timescale 1ns / 1ps

module uart_tx #(
  parameter integer DIV = 33
) (
  input  wire       clk,
  input  wire       rst_n,
  input  wire [7:0] data,
  input  wire       start,
  output reg        tx,
  output reg        busy
);
  localparam integer DW = (DIV <= 2) ? 1 : $clog2(DIV);

  reg [3:0] bit_i;
  reg [DW-1:0] div;
  reg [9:0] sh;

  always @(posedge clk) begin
    if (!rst_n) begin
      tx    <= 1'b1;
      busy  <= 1'b0;
      bit_i <= 4'd0;
      div   <= {DW{1'b0}};
      sh    <= 10'h3FF;
    end else if (!busy) begin
      if (start) begin
        sh    <= {1'b1, data, 1'b0};
        busy  <= 1'b1;
        bit_i <= 4'd0;
        div   <= DIV - 1;
        tx    <= 1'b0;
      end
    end else if (div == 0) begin
      div <= DIV - 1;
      if (bit_i == 4'd9) begin
        busy <= 1'b0;
        tx   <= 1'b1;
      end else begin
        sh    <= {1'b1, sh[9:1]};
        tx    <= sh[1];
        bit_i <= bit_i + 1'b1;
      end
    end else
      div <= div - 1'b1;
  end
endmodule
