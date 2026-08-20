// MMIO wrapper around the existing 16-head D=4 mixer farm.
// Drop this *inside* an AWS F2 CL and map the regs onto OCL AXI-Lite
// (see f2/README.md). This is not a full F2 shell top — the HDK owns that.
//
// Word addresses (byte addr = word << 2), clk = CL clk_main_a0 (~250 MHz):
//   0x000 CTRL    [0]=kick (W1P), [1]=irq enable unused
//   0x001 STATUS  [0]=ready, [1]=done (sticky, W1C via CTRL[2])
//   0x002 CYCLES  wall clocks from kick to done
//   0x040..        packed request, same layout as UART payload (224 B)
//   0x100..        packed response o[192]+cycles already in 0x002 (196 B UART)
`timescale 1ns / 1ps

module cl_qwen_farm_mmio (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        wr_en,
  input  wire        rd_en,
  input  wire [15:0] addr,   // byte address, 4-byte aligned
  input  wire [31:0] wr_data,
  output reg  [31:0] rd_data
);
  localparam integer REQ = 224;
  localparam integer RSP = 196;

  reg [7:0] req [0:REQ-1];
  reg [7:0] rsp [0:RSP-1];

  wire farm_ready, farm_done;
  wire signed [16*96-1:0] o_flat;
  reg farm_en;
  reg [31:0] cycles;
  reg        running;
  reg        done_sticky;

  wire signed [16*32-1:0] q_flat;
  wire signed [16*32-1:0] k_flat;
  wire signed [16*32-1:0] v_flat;
  wire        [16*8-1:0]  g_flat;
  wire        [16*8-1:0]  beta_flat;

  genvar i;
  generate
    for (i = 0; i < 64; i = i + 1) begin : pack
      assign q_flat[i*8 +: 8]    = req[i];
      assign k_flat[i*8 +: 8]    = req[64+i];
      assign v_flat[i*8 +: 8]    = req[128+i];
    end
    for (i = 0; i < 16; i = i + 1) begin : gb
      assign g_flat[i*8 +: 8]    = req[192+i];
      assign beta_flat[i*8 +: 8] = req[208+i];
    end
  endgenerate

  qwen08b_heads16_d4 u_farm (
    .clk(clk),
    .rst_n(rst_n),
    .en(farm_en),
    .q_flat(q_flat),
    .k_flat(k_flat),
    .v_flat(v_flat),
    .g_flat(g_flat),
    .beta_flat(beta_flat),
    .o_flat(o_flat),
    .done(farm_done),
    .ready(farm_ready)
  );

  integer n;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      farm_en <= 1'b0;
      running <= 1'b0;
      done_sticky <= 1'b0;
      cycles <= 32'd0;
      rd_data <= 32'd0;
      for (n = 0; n < REQ; n = n + 1) req[n] <= 8'd0;
      for (n = 0; n < RSP; n = n + 1) rsp[n] <= 8'd0;
    end else begin
      farm_en <= 1'b0;
      if (running) cycles <= cycles + 32'd1;
      if (farm_done && running) begin
        running <= 1'b0;
        done_sticky <= 1'b1;
        for (n = 0; n < 192; n = n + 1)
          rsp[n] <= o_flat[n*8 +: 8];
        rsp[192] <= cycles[7:0];
        rsp[193] <= cycles[15:8];
        rsp[194] <= cycles[23:16];
        rsp[195] <= cycles[31:24];
      end
      if (wr_en) begin
        if (addr[15:2] == 14'd0) begin
          if (wr_data[0] && farm_ready && !running) begin
            farm_en <= 1'b1;
            running <= 1'b1;
            done_sticky <= 1'b0;
            cycles <= 32'd0;
          end
          if (wr_data[2]) done_sticky <= 1'b0;
        end else if (addr[15:8] == 8'h01) begin
          // 0x100.. response is read-only
        end else if (addr[15:8] == 8'h00 && addr[7:2] >= 6'd16) begin
          // 0x040 + : request bytes, 4 per write
          for (n = 0; n < 4; n = n + 1)
            if ((addr - 16'h40 + n) < REQ)
              req[addr - 16'h40 + n] <= wr_data[n*8 +: 8];
        end
      end
      if (rd_en) begin
        case (addr[15:2])
          14'd0: rd_data <= 32'd0;
          14'd1: rd_data <= {30'd0, done_sticky, farm_ready};
          14'd2: rd_data <= cycles;
          default: begin
            if (addr[15:8] == 8'h01) begin
              rd_data <= {rsp[addr-16'h100+3], rsp[addr-16'h100+2],
                          rsp[addr-16'h100+1], rsp[addr-16'h100+0]};
            end else if (addr >= 16'h40 && addr < 16'h40 + REQ) begin
              rd_data <= {req[addr-16'h40+3], req[addr-16'h40+2],
                          req[addr-16'h40+1], req[addr-16'h40+0]};
            end else rd_data <= 32'hDEAD_BEEF;
          end
        endcase
      end
    end
  end
endmodule
