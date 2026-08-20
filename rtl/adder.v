`timescale 1ns / 1ps

module adder (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] a,
    input  wire [7:0] b,
    output reg  [8:0] sum
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sum <= 9'd0;
        else
            sum <= {1'b0, a} + {1'b0, b};
    end
endmodule
