#!/usr/bin/env python3
"""Emit 8x8 spatial-column RTL: bit-serial CSD, via-tap, via-fetch.

All three DUTs hardwire the same W as rtl/ffn_tile_8x8_b4_reg.v.
Golden is integer y = W_int @ x. Stdcell stand-ins — not a via compiler.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pe_xbar.weights import BITS, bit_serial_matvec, load_w  # noqa: E402
from rtl_gen.emit_tile import (  # noqa: E402
    _pack_row_hex,
    emit_add_tree,
    emit_csd_function,
)

IN_W = 8


def _row_params(w: np.ndarray, bits: int) -> str:
    rows, cols = map(int, w.shape)
    lines = []
    names = []
    for r in range(rows):
        name = f"W_ROW{r}"
        names.append(name)
        lines.append(f"  localparam [{cols * bits - 1}:0] {name} = {_pack_row_hex(w[r], bits)};")
    concat = ",\n    ".join(reversed(names))
    lines.append(f"  localparam [{rows * cols * bits - 1}:0] WROM = {{")
    lines.append(f"    {concat}")
    lines.append("  };")
    return "\n".join(lines)


def _header_comment(kind: str, rows: int, cols: int) -> str:
    return "\n".join(
        [
            f"// {kind}",
            f"// {rows}x{cols} 4-bit W. y = W_int @ x (integer).",
            "// Generated; do not edit. pe_xbar/emit.py",
            "`timescale 1ns / 1ps",
            "",
        ]
    )


def emit_tap_function(bits: int) -> str:
    lines = [
        "// Binary two's-complement taps: each weight bit is a via that injects ±(x<<b).",
        f"function automatic signed [IN_W+{bits}-1:0] tap_mul{bits};",
        "  input signed [IN_W-1:0] x;",
        f"  input signed [{bits}-1:0] w;",
        f"  reg signed [IN_W+{bits}-1:0] xs;",
        f"  reg signed [IN_W+{bits}-1:0] acc;",
        "  begin",
        f"    xs = {{{{ {bits}{{x[IN_W-1]}} }}, x}};",
        "    acc = 0;",
    ]
    for b in range(bits - 1):
        lines.append(f"    if (w[{b}]) acc = acc + (xs <<< {b});")
    lines.append(f"    if (w[{bits - 1}]) acc = acc - (xs <<< {bits - 1});")
    lines.append(f"    tap_mul{bits} = acc;")
    lines += ["  end", "endfunction", ""]
    return "\n".join(lines)


def _matvec_generate(mul_name: str, x_name: str = "x") -> str:
    return f"""\
  wire signed [IN_W-1:0] {x_name} [0:COLS-1];
  genvar gi, gj;
  generate
    for (gi = 0; gi < COLS; gi = gi + 1) begin : g_unpack_x
      assign {x_name}[gi] = $signed(x_flat[gi*IN_W +: IN_W]);
    end
  endgenerate

  generate
    for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
      wire signed [ACC_W-1:0] prods [0:COLS-1];
      wire signed [COLS*ACC_W-1:0] prods_flat;
      wire signed [ACC_W-1:0] acc;
      for (gj = 0; gj < COLS; gj = gj + 1) begin : g_col
        wire signed [IN_W+W_W-1:0] p_raw;
        assign p_raw = {mul_name}({x_name}[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
        assign prods[gj] = {{{{ (ACC_W-(IN_W+W_W)){{p_raw[IN_W+W_W-1]}} }}, p_raw}};
        assign prods_flat[gj*ACC_W +: ACC_W] = prods[gj];
      end
      add_tree #(.N(COLS), .W(ACC_W)) u_tree (
        .xs(prods_flat),
        .y(acc)
      );
      assign y_flat[gi*ACC_W +: ACC_W] = acc;
    end
  endgenerate
"""


def emit_rom_tap(w: np.ndarray, bits: int) -> str:
    rows, cols = map(int, w.shape)
    acc_w = IN_W + bits + int(math.ceil(math.log2(max(cols, 2))))
    body = [
        _header_comment("Via-tap mat-vec: x broadcast, each weight bit is a digital via (AND/shift).", rows, cols),
        emit_add_tree(),
        "module ffn_rom_tap #(",
        f"  parameter integer ROWS = {rows},",
        f"  parameter integer COLS = {cols},",
        f"  parameter integer IN_W = {IN_W},",
        f"  parameter integer W_W  = {bits},",
        f"  parameter integer ACC_W = {acc_w}",
        ") (",
        "  input  wire signed [COLS*IN_W-1:0] x_flat,",
        "  output wire signed [ROWS*ACC_W-1:0] y_flat",
        ");",
        _row_params(w, bits),
        "",
        emit_tap_function(bits),
        _matvec_generate(f"tap_mul{bits}", "x"),
        "endmodule",
        "",
        "// Registered wrapper for STA / PnR.",
        "module ffn_rom_tap_reg (",
        "  input  wire clk,",
        f"  input  wire signed [{cols * IN_W}-1:0] x_flat,",
        f"  output reg  signed [{rows * acc_w}-1:0] y_flat",
        ");",
        f"  reg  signed [{cols * IN_W}-1:0] x_q;",
        f"  wire signed [{rows * acc_w}-1:0] y_c;",
        "  ffn_rom_tap u_comb (",
        "    .x_flat(x_q),",
        "    .y_flat(y_c)",
        "  );",
        "  always @(posedge clk) begin",
        "    x_q    <= x_flat;",
        "    y_flat <= y_c;",
        "  end",
        "endmodule",
        "",
    ]
    return "\n".join(body)


def emit_col_serial(w: np.ndarray, bits: int) -> str:
    rows, cols = map(int, w.shape)
    acc_w = IN_W + bits + int(math.ceil(math.log2(max(cols, 2))))
    return f"""\
{_header_comment("Bit-serial CSD column: 8 phases of x bits, CSD taps, digital partials.", rows, cols)}
{emit_add_tree()}
module ffn_col_serial #(
  parameter integer ROWS = {rows},
  parameter integer COLS = {cols},
  parameter integer IN_W = {IN_W},
  parameter integer W_W  = {bits},
  parameter integer ACC_W = {acc_w}
) (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output reg  signed [ROWS*ACC_W-1:0] y_flat,
  output reg                       done
);
{_row_params(w, bits)}

{emit_csd_function(bits, IN_W)}
  localparam ST_IDLE = 1'b0;
  localparam ST_RUN  = 1'b1;

  reg        st;
  reg  [2:0] k;
  reg signed [IN_W-1:0]  x_r  [0:COLS-1];
  reg signed [31:0]      acc  [0:ROWS-1];

  wire signed [IN_W-1:0] xb [0:COLS-1];
  genvar gx;
  generate
    for (gx = 0; gx < COLS; gx = gx + 1) begin : g_xb
      assign xb[gx] = {{{{ (IN_W-1){{1'b0}} }}, x_r[gx][k]}};
    end
  endgenerate

  wire signed [ACC_W-1:0] part [0:ROWS-1];
  wire signed [31:0]      contrib [0:ROWS-1];
  wire signed [ROWS*ACC_W-1:0] y_next_flat;
  genvar gi, gj;
  generate
    for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row
      wire signed [ACC_W-1:0] prods [0:COLS-1];
      wire signed [COLS*ACC_W-1:0] prods_flat;
      for (gj = 0; gj < COLS; gj = gj + 1) begin : g_col
        wire signed [IN_W+W_W-1:0] p_raw;
        assign p_raw = csd_mul{bits}(xb[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));
        assign prods[gj] = {{{{ (ACC_W-(IN_W+W_W)){{p_raw[IN_W+W_W-1]}} }}, p_raw}};
        assign prods_flat[gj*ACC_W +: ACC_W] = prods[gj];
      end
      add_tree #(.N(COLS), .W(ACC_W)) u_tree (
        .xs(prods_flat),
        .y(part[gi])
      );
      wire signed [31:0] part_ext = {{{{ (32-ACC_W){{part[gi][ACC_W-1]}} }}, part[gi]}};
      assign contrib[gi] = (k == 3'd7) ? -(part_ext <<< k) : (part_ext <<< k);
      wire signed [ACC_W-1:0] y_next = acc[gi] + contrib[gi];
      assign y_next_flat[gi*ACC_W +: ACC_W] = y_next;
    end
  endgenerate

  integer ci, ri;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      k <= 3'd0;
      y_flat <= {{ROWS * ACC_W{{1'b0}}}};
      for (ci = 0; ci < COLS; ci = ci + 1)
        x_r[ci] <= {{IN_W{{1'b0}}}};
      for (ri = 0; ri < ROWS; ri = ri + 1)
        acc[ri] <= 32'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (ci = 0; ci < COLS; ci = ci + 1)
              x_r[ci] <= $signed(x_flat[ci*IN_W +: IN_W]);
            for (ri = 0; ri < ROWS; ri = ri + 1)
              acc[ri] <= 32'sd0;
            k <= 3'd0;
            st <= ST_RUN;
          end
        end
        ST_RUN: begin
          for (ri = 0; ri < ROWS; ri = ri + 1)
            acc[ri] <= acc[ri] + contrib[ri];
          if (k == 3'd7) begin
            y_flat <= y_next_flat;
            done <= 1'b1;
            st <= ST_IDLE;
          end else
            k <= k + 3'd1;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
"""


def emit_rom_fetch(w: np.ndarray, bits: int) -> str:
    rows, cols = map(int, w.shape)
    acc_w = IN_W + bits + int(math.ceil(math.log2(max(cols, 2))))
    return f"""\
{_header_comment("Via-ROM as memory: decode one (row,col), one MAC, scan the matrix.", rows, cols)}
module ffn_rom_fetch #(
  parameter integer ROWS = {rows},
  parameter integer COLS = {cols},
  parameter integer IN_W = {IN_W},
  parameter integer W_W  = {bits},
  parameter integer ACC_W = {acc_w}
) (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire                      en,
  input  wire signed [COLS*IN_W-1:0] x_flat,
  output reg  signed [ROWS*ACC_W-1:0] y_flat,
  output reg                       done
);
{_row_params(w, bits)}

  localparam ST_IDLE = 2'd0;
  localparam ST_MAC  = 2'd1;
  localparam ST_PACK = 2'd2;

  reg  [1:0] st;
  reg  [2:0] i;
  reg  [2:0] j;
  reg signed [IN_W-1:0] x_r [0:COLS-1];
  reg signed [31:0]     acc [0:ROWS-1];

  wire [5:0] idx = {{i, j}};
  wire signed [W_W-1:0] w_sel = $signed(WROM[idx*W_W +: W_W]);
  wire signed [31:0]    prod  = $signed(w_sel) * $signed(x_r[j]);

  integer ci, ri;
  always @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      st <= ST_IDLE;
      i <= 3'd0;
      j <= 3'd0;
      y_flat <= {{ROWS * ACC_W{{1'b0}}}};
      for (ci = 0; ci < COLS; ci = ci + 1)
        x_r[ci] <= {{IN_W{{1'b0}}}};
      for (ri = 0; ri < ROWS; ri = ri + 1)
        acc[ri] <= 32'sd0;
    end else begin
      case (st)
        ST_IDLE: begin
          if (en) begin
            for (ci = 0; ci < COLS; ci = ci + 1)
              x_r[ci] <= $signed(x_flat[ci*IN_W +: IN_W]);
            for (ri = 0; ri < ROWS; ri = ri + 1)
              acc[ri] <= 32'sd0;
            i <= 3'd0;
            j <= 3'd0;
            st <= ST_MAC;
          end
        end
        ST_MAC: begin
          acc[i] <= acc[i] + prod;
          if (j == 3'd7) begin
            j <= 3'd0;
            if (i == 3'd7)
              st <= ST_PACK;
            else
              i <= i + 3'd1;
          end else
            j <= j + 3'd1;
        end
        ST_PACK: begin
          for (ri = 0; ri < ROWS; ri = ri + 1)
            y_flat[ri*ACC_W +: ACC_W] <= acc[ri][ACC_W-1:0];
          done <= 1'b1;
          st <= ST_IDLE;
        end
        default: st <= ST_IDLE;
      endcase
    end
  end
endmodule
"""


def write_all(out_dir: Path, w: np.ndarray, bits: int, kinds: tuple[str, ...] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = map(int, w.shape)
    suffix = "" if (rows, cols) == (8, 8) else f"_{rows}x{cols}_b{bits}"
    kind_map = {
        "serial": (f"ffn_col_serial{suffix}.v", emit_col_serial(w, bits)),
        "tap": (f"ffn_rom_tap{suffix}.v", emit_rom_tap(w, bits)),
        "fetch": (f"ffn_rom_fetch{suffix}.v", emit_rom_fetch(w, bits)),
    }
    if kinds is None:
        kinds = ("serial", "tap", "fetch") if rows == 8 else ("serial",)
    written = []
    for kind in kinds:
        name, text = kind_map[kind]
        path = out_dir / name
        path.write_text(text)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "rtl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--kinds", default="", help="comma: serial,tap,fetch (default: all at n=8, serial otherwise)")
    args = parser.parse_args()
    w = load_w(args.n, BITS)
    x = (np.arange(args.n, dtype=np.int32) % 17) - 8
    y = bit_serial_matvec(w, x)
    if not np.array_equal(y, w.astype(np.int32) @ x):
        raise SystemExit("bit-serial golden != W @ x")
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip()) or None
    paths = write_all(args.out_dir, w, BITS, kinds)
    npy = ROOT / "artifacts" / (f"tile{args.n}_int4_xbar.npy" if args.n != 8 else "tile8_int4_xbar.npy")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, w)
    nz = int(np.count_nonzero(w))
    print(f"W {args.n}x{args.n} 4-bit nonzero={nz}/{w.size}  saved {npy}")
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
