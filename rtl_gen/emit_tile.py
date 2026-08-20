"""Emit Verilog for a hardwired CSD mat-vec tile."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtl_gen.csd import signed_range, verilog_csd_expr


def _twos(v: int, bits: int) -> int:
    return int(v) & ((1 << bits) - 1)


def _pack_row_hex(row: np.ndarray, bits: int) -> str:
    acc = 0
    for i, v in enumerate(row.tolist()):
        acc |= _twos(v, bits) << (i * bits)
    width = len(row) * bits
    nibbles = (width + 3) // 4
    return f"{width}'h{acc:0{nibbles}x}"


def emit_csd_function(bits: int, in_w: int = 8) -> str:
    """CSD shift-add as a function: synth const-folds each call; sim stays flat."""
    lines = [
        "// Constant-coefficient multiply: CSD recoding → shift-add/sub, never a general *.",
        f"// Signed {bits}-bit symmetric range {min(signed_range(bits))}..{max(signed_range(bits))}.",
        "// Each unique weight maps to one expression (inlined at each call site).",
        f"function automatic signed [IN_W+{bits}-1:0] csd_mul{bits};",
        "  input signed [IN_W-1:0] x;",
        f"  input signed [{bits}-1:0] w;",
        f"  reg signed [IN_W+{bits}-1:0] xs;",
        "  begin",
        f"    xs = {{{{ {bits}{{x[IN_W-1]}} }}, x}};",
        "    case (w)",
    ]
    for w in signed_range(bits):
        expr = verilog_csd_expr(w, "xs")
        lines.append(f"      {bits}'h{_twos(w, bits):x}: csd_mul{bits} = {expr};  // {w}")
    lines += [
        f"      default: csd_mul{bits} = 0;",
        "    endcase",
        "  end",
        "endfunction",
        "",
    ]
    return "\n".join(lines)


def emit_add_tree() -> str:
    return """\
// N-input signed adder. The loop is a reduction; synth maps it to a tree.
module add_tree #(
  parameter integer N = 128,
  parameter integer W = 20
) (
  input  wire signed [N*W-1:0] xs,
  output reg  signed [W-1:0]   y
);
  integer k;
  reg signed [W-1:0] acc;
  always @* begin
    acc = {W{1'b0}};
    for (k = 0; k < N; k = k + 1)
      acc = acc + $signed(xs[k*W +: W]);
    y = acc;
  end
endmodule
"""


def emit_registered_wrapper(
    bits: int,
    in_w: int,
    rows: int,
    cols: int,
    comb_module: str = "ffn_tile",
    top: str = "ffn_tile_reg",
) -> str:
    acc_w = in_w + bits + int(np.ceil(np.log2(max(cols, 2))))
    return f"""\
// Registered wrapper for STA / PnR. Combo core is `{comb_module}`.
module {top} (
  input  wire clk,
  input  wire signed [{cols * in_w}-1:0] x_flat,
  output reg  signed [{rows * acc_w}-1:0] y_flat
);
  reg  signed [{cols * in_w}-1:0] x_q;
  wire signed [{rows * acc_w}-1:0] y_c;
  {comb_module} u_comb (
    .x_flat(x_q),
    .y_flat(y_c)
  );
  always @(posedge clk) begin
    x_q    <= x_flat;
    y_flat <= y_c;
  end
endmodule
"""


def emit_tile(
    w_int: np.ndarray,
    bits: int,
    in_w: int = 8,
    module: str = "ffn_tile",
    registered: bool = False,
    reg_module: str = "ffn_tile_reg",
) -> str:
    rows, cols = map(int, w_int.shape)
    acc_w = in_w + bits + int(np.ceil(np.log2(max(cols, 2))))
    w_int = np.asarray(w_int, dtype=np.int8)

    row_params: list[str] = []
    concat_names: list[str] = []
    for r in range(rows):
        name = f"W_ROW{r}"
        row_params.append(f"  localparam [{cols * bits - 1}:0] {name} = {_pack_row_hex(w_int[r], bits)};")
        concat_names.append(name)

    wrom_concat = ",\n    ".join(reversed(concat_names))

    header = [
        f"// Hardwired mat-vec tile {rows}x{cols}, signed {bits}-bit weights.",
        "// y = W_int * x  (integer; per-row scale is outside this module).",
        "// Each weight is a CSD shift-add (csd_mul*), not a general multiplier.",
        "// Generated from Qwen3.5-0.8B layer0 down_proj (per-output-row quant).",
        "`timescale 1ns / 1ps",
        "",
        emit_add_tree(),
        f"module {module} #(",
        f"  parameter integer ROWS = {rows},",
        f"  parameter integer COLS = {cols},",
        f"  parameter integer IN_W = {in_w},",
        f"  parameter integer W_W  = {bits},",
        f"  parameter integer ACC_W = {acc_w}",
        ") (",
        "  input  wire signed [COLS*IN_W-1:0] x_flat,",
        "  output wire signed [ROWS*ACC_W-1:0] y_flat",
        ");",
        *row_params,
        f"  localparam [{rows * cols * bits - 1}:0] WROM = {{",
        f"    {wrom_concat}",
        "  };",
        "",
        emit_csd_function(bits, in_w),
        "  wire signed [IN_W-1:0] x [0:COLS-1];",
        "  genvar gi, gj;",
        "  generate",
        "    for (gi = 0; gi < COLS; gi = gi + 1) begin : g_unpack_x",
        "      assign x[gi] = $signed(x_flat[gi*IN_W +: IN_W]);",
        "    end",
        "  endgenerate",
        "",
        "  generate",
        "    for (gi = 0; gi < ROWS; gi = gi + 1) begin : g_row",
        "      wire signed [ACC_W-1:0] prods [0:COLS-1];",
        "      wire signed [COLS*ACC_W-1:0] prods_flat;",
        "      wire signed [ACC_W-1:0] acc;",
        "      for (gj = 0; gj < COLS; gj = gj + 1) begin : g_col",
        "        wire signed [IN_W+W_W-1:0] p_raw;",
        f"        assign p_raw = csd_mul{bits}(x[gj], $signed(WROM[(gi*COLS+gj)*W_W +: W_W]));",
        "        assign prods[gj] = {{(ACC_W-(IN_W+W_W)){p_raw[IN_W+W_W-1]}}, p_raw};",
        "        assign prods_flat[gj*ACC_W +: ACC_W] = prods[gj];",
        "      end",
        "      add_tree #(.N(COLS), .W(ACC_W)) u_tree (",
        "        .xs(prods_flat),",
        "        .y(acc)",
        "      );",
        "      assign y_flat[gi*ACC_W +: ACC_W] = acc;",
        "    end",
        "  endgenerate",
        "endmodule",
        "",
    ]
    body = "\n".join(header)
    if registered:
        rows, cols = map(int, w_int.shape)
        body += emit_registered_wrapper(bits, in_w, rows, cols, comb_module=module, top=reg_module)
    return body


def write_tile(
    path: Path,
    w_int: np.ndarray,
    bits: int,
    in_w: int = 8,
    module: str = "ffn_tile",
    registered: bool = False,
    reg_module: str = "ffn_tile_reg",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        emit_tile(w_int, bits, in_w=in_w, module=module, registered=registered, reg_module=reg_module)
    )
