// Q8 exp LUT: 256 * exp(-k/8) for k = (m - s) in integer score units.
// Matches quant/attn_online_int.py EXP_TAB. 9 bits so 256 fits.
`timescale 1ns / 1ps

module attn_exp_lut (
  input  wire [5:0] d,
  output reg  [8:0] y
);
  always @* begin
    case (d)
      6'd0:  y = 9'd256;
      6'd1:  y = 9'd226;
      6'd2:  y = 9'd199;
      6'd3:  y = 9'd176;
      6'd4:  y = 9'd155;
      6'd5:  y = 9'd137;
      6'd6:  y = 9'd121;
      6'd7:  y = 9'd107;
      6'd8:  y = 9'd94;
      6'd9:  y = 9'd83;
      6'd10: y = 9'd73;
      6'd11: y = 9'd65;
      6'd12: y = 9'd57;
      6'd13: y = 9'd50;
      6'd14: y = 9'd44;
      6'd15: y = 9'd39;
      6'd16: y = 9'd35;
      6'd17: y = 9'd31;
      6'd18: y = 9'd27;
      6'd19: y = 9'd24;
      6'd20: y = 9'd21;
      6'd21: y = 9'd19;
      6'd22: y = 9'd16;
      6'd23: y = 9'd14;
      6'd24: y = 9'd13;
      6'd25: y = 9'd11;
      6'd26: y = 9'd10;
      6'd27: y = 9'd9;
      6'd28: y = 9'd8;
      6'd29: y = 9'd7;
      6'd30: y = 9'd6;
      6'd31: y = 9'd5;
      6'd32: y = 9'd5;
      default: y = 9'd0;
    endcase
  end
endmodule
