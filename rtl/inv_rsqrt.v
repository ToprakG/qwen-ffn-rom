// Combo Q16 inverse-sqrt: inv ≈ (1<<16) / sqrt(x).
// CLZ normalize → 64-entry LUT seed → one Newton-Raphson step.
// Bit-exact vs quant/rsqrt_int.py inv_rsqrt_q16.
`timescale 1ns / 1ps

module inv_rsqrt (
  input  wire [31:0] x,
  output wire [16:0] inv
);
  function automatic [5:0] clz32;
    input [31:0] a;
    reg [5:0]  n;
    reg [31:0] t;
    begin
      t = a;
      n = 6'd0;
      if (t[31:16] == 16'd0) begin n = n + 6'd16; t = {t[15:0], 16'd0}; end
      if (t[31:24] ==  8'd0) begin n = n + 6'd8;  t = {t[23:0],  8'd0}; end
      if (t[31:28] ==  4'd0) begin n = n + 6'd4;  t = {t[27:0],  4'd0}; end
      if (t[31:30] ==  2'd0) begin n = n + 6'd2;  t = {t[29:0],  2'd0}; end
      if (t[31]    ==  1'b0) n = n + 6'd1;
      clz32 = (a == 32'd0) ? 6'd32 : n;
    end
  endfunction

  wire [31:0] x1      = (x == 32'd0) ? 32'd1 : x;
  wire [5:0]  lz      = clz32(x1);
  wire [5:0]  sh      = {lz[5:1], 1'b0};
  wire [31:0] xn      = x1 << sh[4:0];
  wire [5:0]  idx     = xn[31:26];
  wire [18:0] seed;
  rsqrt_lut u_lut (.idx(idx), .y(seed));

  wire [37:0] y2      = seed * seed;
  wire [69:0] xyy     = xn * y2;
  wire [23:0] xyy_q16 = xyy[69:46];
  wire [18:0] inner   = (xyy_q16 > 24'd196608) ? 19'd0
                      : (19'd196608 - xyy_q16[18:0]);
  wire [37:0] yprod   = seed * inner;
  wire [20:0] y_n     = yprod[37:17];
  wire [4:0]  rsh     = 5'd15 - sh[5:1];
  wire [20:0] inv_s   = y_n >> rsh;
  assign inv = (inv_s == 21'd0) ? 17'd1
             : (inv_s > 21'd131071) ? 17'd131071
             : inv_s[16:0];
endmodule
