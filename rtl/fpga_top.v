// Nexys Video (or any 7-series) top: UART host around the 16-head D=4 farm.
// Protocol:  host → 0xA5 || q[64] || k[64] || v[64] || g[16] || beta[16]
//            FPGA → 0x5A || o[192] || cycles[4]   (cycles little-endian uint32)
`timescale 1ns / 1ps

module fpga_top #(
  parameter integer CLK_HZ   = 100_000_000,
  parameter integer BAUD     = 3_000_000
) (
  input  wire clk,
  input  wire cpu_reset,
  input  wire uart_rx,
  output wire uart_tx,
  output wire led_heartbeat,
  output wire led_ready,
  output wire led_done
);
  localparam integer DIV = (CLK_HZ / BAUD) < 2 ? 2 : (CLK_HZ / BAUD);
  localparam integer REQ = 224;
  localparam integer RSP = 196;

  wire rst_n = ~cpu_reset;

  wire [7:0] rx_data;
  wire       rx_valid;
  wire       tx_busy;
  reg  [7:0] tx_data;
  reg        tx_start;

  uart_rx #(.DIV(DIV)) u_rx (
    .clk(clk), .rst_n(rst_n), .rx(uart_rx), .data(rx_data), .valid(rx_valid)
  );
  uart_tx #(.DIV(DIV)) u_tx (
    .clk(clk), .rst_n(rst_n), .data(tx_data), .start(tx_start), .tx(uart_tx), .busy(tx_busy)
  );

  reg signed [16*32-1:0] q_flat;
  reg signed [16*32-1:0] k_flat;
  reg signed [16*32-1:0] v_flat;
  reg        [16*8-1:0]  g_flat;
  reg        [16*8-1:0]  beta_flat;
  wire signed [16*96-1:0] o_flat;
  wire farm_done;
  wire farm_ready;
  reg  farm_en;

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

  localparam [2:0] ST_WAIT_M = 3'd0;
  localparam [2:0] ST_RX     = 3'd1;
  localparam [2:0] ST_LOAD   = 3'd2;
  localparam [2:0] ST_KICK   = 3'd3;
  localparam [2:0] ST_RUN    = 3'd4;
  localparam [2:0] ST_TX_M   = 3'd5;
  localparam [2:0] ST_TX     = 3'd6;

  reg [2:0]  st;
  reg [7:0]  rx_buf [0:REQ-1];
  reg [7:0]  tx_buf [0:RSP-1];
  reg [7:0]  rx_i;
  reg [7:0]  tx_i;
  reg [31:0] cycles;
  reg        running;
  reg        done_sticky;
  reg [23:0] hb;

  integer n;

  assign led_heartbeat = hb[23];
  assign led_ready     = farm_ready;
  assign led_done      = done_sticky;

  always @(posedge clk) begin
    farm_en  <= 1'b0;
    tx_start <= 1'b0;
    hb       <= hb + 1'b1;
    if (!rst_n) begin
      st          <= ST_WAIT_M;
      rx_i        <= 8'd0;
      tx_i        <= 8'd0;
      cycles      <= 32'd0;
      running     <= 1'b0;
      done_sticky <= 1'b0;
      q_flat      <= {16*32{1'b0}};
      k_flat      <= {16*32{1'b0}};
      v_flat      <= {16*32{1'b0}};
      g_flat      <= {16*8{1'b0}};
      beta_flat   <= {16*8{1'b0}};
    end else begin
      if (running)
        cycles <= cycles + 1'b1;
      if (farm_done)
        running <= 1'b0;

      case (st)
        ST_WAIT_M: begin
          rx_i <= 8'd0;
          if (rx_valid && rx_data == 8'hA5)
            st <= ST_RX;
        end
        ST_RX: begin
          if (rx_valid) begin
            rx_buf[rx_i] <= rx_data;
            if (rx_i == REQ - 1)
              st <= ST_LOAD;
            else
              rx_i <= rx_i + 1'b1;
          end
        end
        ST_LOAD: begin
          for (n = 0; n < 64; n = n + 1) begin
            q_flat[n*8 +: 8] <= rx_buf[n];
            k_flat[n*8 +: 8] <= rx_buf[64 + n];
            v_flat[n*8 +: 8] <= rx_buf[128 + n];
          end
          for (n = 0; n < 16; n = n + 1) begin
            g_flat[n*8 +: 8]    <= rx_buf[192 + n];
            beta_flat[n*8 +: 8] <= rx_buf[208 + n];
          end
          st <= ST_KICK;
        end
        ST_KICK: begin
          if (farm_ready) begin
            farm_en  <= 1'b1;
            cycles   <= 32'd0;
            running  <= 1'b1;
            st       <= ST_RUN;
          end
        end
        ST_RUN: begin
          if (farm_done) begin
            for (n = 0; n < 192; n = n + 1)
              tx_buf[n] <= o_flat[n*8 +: 8];
            tx_buf[192] <= cycles[7:0];
            tx_buf[193] <= cycles[15:8];
            tx_buf[194] <= cycles[23:16];
            tx_buf[195] <= cycles[31:24];
            done_sticky <= 1'b1;
            st          <= ST_TX_M;
          end
        end
        ST_TX_M: begin
          if (!tx_busy) begin
            tx_data  <= 8'h5A;
            tx_start <= 1'b1;
            tx_i     <= 8'd0;
            st       <= ST_TX;
          end
        end
        ST_TX: begin
          if (!tx_busy && !tx_start) begin
            tx_data  <= tx_buf[tx_i];
            tx_start <= 1'b1;
            if (tx_i == RSP - 1)
              st <= ST_WAIT_M;
            else
              tx_i <= tx_i + 1'b1;
          end
        end
        default: st <= ST_WAIT_M;
      endcase
    end
  end
endmodule
