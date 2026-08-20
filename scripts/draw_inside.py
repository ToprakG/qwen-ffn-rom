#!/usr/bin/env python3
"""Datapath insides: FFN via-tap (HC1-style) and fused DeltaNet column PE.

  python scripts/draw_inside.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyBboxPatch,
    FancyArrowPatch,
    Polygon,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

INK = "#1a1a1a"
MUTED = "#4a4a4a"
WCOL = "#d6e4f0"       # weight / SRAM
WEC = "#2c5f8a"
PCOL = "#f3d4dc"       # pre-multiply
PEC = "#b54a6a"
SCOL = "#f7e8b0"       # selection / vias
SEC = "#b08900"
ACOL = "#d5ead7"       # accumulate
AEC = "#2d7a4f"
WHITE = "#ffffff"
VIA_ON = "#1a1a1a"
VIA_OFF = "#c8c8c8"


def region(ax, x, y, w, h, fc, ec, tag, title):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.6, zorder=0, alpha=0.95))
    ax.text(x + 0.08, y + h - 0.12, tag, fontsize=9, fontweight="bold",
            color=ec, va="top", zorder=4)
    ax.text(x + 0.42, y + h - 0.12, title, fontsize=8, color=INK,
            va="top", zorder=4)


def blk(ax, x, y, w, h, text, *, fc=WHITE, ec=INK, fs=7.5, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.25)


def wire(ax, x0, y0, x1, y1, *, c=INK, lw=1.0, z=1):
    ax.plot([x0, x1], [y0, y1], color=c, lw=lw, zorder=z, solid_capstyle="round")


def arr(ax, x0, y0, x1, y1, *, c=INK, lw=1.15):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=9, lw=lw, color=c, zorder=3,
    ))


def via(ax, x, y, on=True, r=0.055):
    ax.add_patch(Circle((x, y), r, facecolor=VIA_ON if on else WHITE,
                        edgecolor=INK, linewidth=0.9, zorder=5))
    if not on:
        ax.add_patch(Circle((x, y), r * 0.35, facecolor=VIA_OFF,
                            edgecolor="none", zorder=6))


def mux(ax, x, y, w, h, label="MUX"):
    # trapezoid pointing right
    pts = [(x, y), (x, y + h), (x + w, y + h * 0.72), (x + w, y + h * 0.28)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=WHITE, edgecolor=INK,
                         linewidth=1.1, zorder=2))
    ax.text(x + w * 0.42, y + h / 2, label, ha="center", va="center",
            fontsize=6.5, zorder=3)


def tree(ax, x, y, w, h, n=3):
    """Tiny adder tree: n leaves into + boxes."""
    blk(ax, x, y, w, h, "+\nadd tree", fc=WHITE, ec=AEC, fs=7)


def draw_ffn(ax):
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 8.15)
    ax.axis("off")
    ax.set_title(
        "FFN via-tap   (one 8×8 tile shown as 4 columns × 3 rows)    "
        "rtl/ffn_rom_tap.v",
        loc="left", fontsize=11, fontweight="bold", color=INK, pad=6,
    )

    # regions: weight | premul | select | accum
    region(ax, 0.15, 0.35, 2.55, 7.15, WCOL, WEC, "a", "Weight  ·  via-ROM")
    region(ax, 2.85, 0.35, 3.35, 7.15, PCOL, PEC, "b", "Pre-multiply  ·  one x")
    region(ax, 6.35, 0.35, 6.15, 7.15, SCOL, SEC, "c", "Selection  ·  vias are the weights")
    region(ax, 12.65, 0.35, 3.35, 7.15, ACOL, AEC, "d", "Accumulate")

    # --- a: ROM cells ---
    ax.text(1.42, 6.95, "4 vias / weight", ha="center", fontsize=6.5, color=WEC)
    for r, bits in enumerate(["1101", "0010", "1001"]):
        yy = 5.85 - r * 1.55
        blk(ax, 0.38, yy, 2.1, 1.25,
            f"W[{r},·]  Q4\n{bits}…  (row {r})",
            fc=WHITE, ec=WEC, fs=7)
    blk(ax, 0.38, 0.55, 2.1, 1.15, "WL decode\nrow addr", fc=WHITE, ec=WEC, fs=7)
    arr(ax, 1.43, 0.55, 1.43, 0.35)
    ax.text(1.43, 0.18, "baked at tapeout\nnot an SRAM read", ha="center",
            fontsize=6.4, color=MUTED, va="top")

    # --- b: x → four rails ---
    blk(ax, 3.15, 6.55, 2.75, 0.55, "x  (int8, broadcast down the column)",
        fc=WHITE, ec=PEC, fs=6.8)
    rails = [
        (5.9, r"$x$"),
        (4.85, r"$x \ll 1$"),
        (3.80, r"$x \ll 2$"),
        (2.75, r"$-x \ll 3$"),
    ]
    # four shift boxes
    labels = [r"$\times 1$", r"$\times 2$", r"$\times 4$", r"$\times(-8)$"]
    for i, lab in enumerate(labels):
        yy = 5.55 - i * 1.05
        blk(ax, 3.35, yy, 2.35, 0.72, lab, fc=WHITE, ec=PEC, fs=8)
        arr(ax, 4.52, 6.55, 4.52, yy + 0.72)
        wire(ax, 5.70, yy + 0.36, 6.55, yy + 0.36, c=PEC, lw=1.3)
    ax.text(4.52, 1.35, "one column of x\nreplicated 8× in the tile\n5120 cols on the chip",
            ha="center", fontsize=6.5, color=PEC, va="top")

    # --- c: via crossbar ---
    # vertical rails from premul, 4 of them, then 4 columns of cells
    rail_y = [5.55 + 0.36 - i * 1.05 for i in range(4)]  # 5.91, 4.86, 3.81, 2.76
    col_x = [7.05, 8.45, 9.85, 11.25]
    # extend verticals
    for ry in rail_y:
        wire(ax, 6.55, ry, 12.35, ry, c=PEC, lw=1.05, z=1)

    # row horizontals
    row_y = [6.35, 4.80, 3.25]
    for ry in row_y:
        wire(ax, 6.85, ry, 12.45, ry, c=INK, lw=1.25, z=1)

    # via pattern per (row, col): 4 bits on the 4 rails. Example weights.
    # row0: 1101 → rails 0,1,3 (x, x<<1, -x<<3)  indices 0=x,1=<<1,2=<<2,3=-<<3
    pattern = [
        [[1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
        [[0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]],
        [[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
    ]
    for ri, ry in enumerate(row_y):
        for ci, cx in enumerate(col_x):
            for bi, raily in enumerate(rail_y):
                via(ax, cx, raily, on=bool(pattern[ri][ci][bi]))
            # tap from rails down to row line at this column
            wire(ax, cx, min(rail_y), cx, ry, c=INK, lw=0.7, z=1)

    ax.text(9.15, 6.88, "via present  =  that shift is added into the row",
            ha="center", fontsize=6.6, color=SEC)
    ax.text(9.15, 2.35,
            "empty hole  =  0  (cell still occupies pitch)\n"
            "this grid × 17.1B weights  is the 1,753 mm²",
            ha="center", fontsize=6.6, color=MUTED, va="top")

    # --- d: adder trees ---
    for i, ry in enumerate(row_y):
        arr(ax, 12.45, ry, 12.85, ry, c=AEC)
        blk(ax, 12.90, ry - 0.38, 2.85, 0.76, f"+  tree  →  y[{i}]",
            fc=WHITE, ec=AEC, fs=7.5)
    ax.text(14.30, 2.35, "8-input add_tree per row\nchip: 5120-wide farm\nSiLU folds here (0 extra clk)",
            ha="center", fontsize=6.5, color=AEC, va="top")

    ax.text(0.15, 0.08,
            "y[r] = Σ_c  (  W[r,c][0]·x  +  W[r,c][1]·(x≪1)  +  W[r,c][2]·(x≪2)  −  W[r,c][3]·(x≪3)  )     "
            "Q4 two’s-complement tap, rtl/ffn_rom_tap.v tap_mul4",
            fontsize=7.0, color=MUTED, va="bottom")


def draw_delta(ax):
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 8.15)
    ax.axis("off")
    ax.set_title(
        "DeltaNet fused PE   (one column j of S, D=128 lanes)    "
        "rtl/gated_delta_fused.v   cycles = D+2 = 130",
        loc="left", fontsize=11, fontweight="bold", color=INK, pad=6,
    )

    region(ax, 0.15, 0.35, 3.35, 7.15, WCOL, WEC, "a", "State  ·  S SRAM  (not via-ROM)")
    region(ax, 3.65, 0.35, 4.55, 7.15, PCOL, PEC, "b", "Pre-multiply  ·  4 muls / lane")
    region(ax, 8.35, 0.35, 3.55, 7.15, SCOL, SEC, "c", "Column select  ·  1R1W")
    region(ax, 12.05, 0.35, 3.95, 7.15, ACOL, AEC, "d", "Accumulate + writeback")

    # --- a: S banks ---
    blk(ax, 0.40, 6.35, 2.85, 0.70, "S  ∈  ℤ¹⁶   D×D per V-head", fc=WHITE, ec=WEC, fs=7)
    for i, name in enumerate(["bank 0  S[0, :]", "bank 1  S[1, :]", "bank D-1 S[D-1,:]"]):
        yy = 5.35 - i * 1.15
        blk(ax, 0.45, yy, 2.75, 0.85, name + "\n1R1W compiler macro",
            fc=WHITE, ec=WEC, fs=6.6)
    blk(ax, 0.45, 0.55, 2.75, 0.95, "addr = column j\nread S[:, j]  this cycle\nwrite S'[:, j] next",
        fc=WHITE, ec=WEC, fs=6.6)
    arr(ax, 3.20, 1.02, 3.65, 1.02, c=WEC)
    ax.text(1.82, 7.05, "48 V-heads × 48 layers\n75.5 MB int16  →  77 mm² SRAM",
            ha="center", fontsize=6.4, color=WEC, va="top")

    # --- b: lane datapath ---
    blk(ax, 3.90, 6.45, 4.05, 0.62, "lane t  (t = 0 … 127 in parallel)",
        fc=WHITE, ec=PEC, fs=7)
    steps = [
        (5.35, r"$s_{\mathrm{dec}} = (S[:,j] \cdot g) \gg 8$"),
        (4.15, r"$p_k = s_{\mathrm{dec}} \cdot k$"),
        (2.95, r"$s' = \mathrm{sat}(s_{\mathrm{dec}} + (k \cdot \delta)\gg 8)$"),
        (1.75, r"$p_q = s' \cdot q$"),
    ]
    for yy, lab in steps:
        blk(ax, 3.90, yy, 4.05, 0.95, lab, fc=WHITE, ec=PEC, fs=8)
    for y0, y1 in ((6.45, 6.30), (5.35, 5.10), (4.15, 3.90), (2.95, 2.70)):
        arr(ax, 5.92, y0, 5.92, y1, c=PEC)

    ax.text(5.92, 1.45, "g, β, q, k, v  are PE inputs\n(not 17B of vias)",
            ha="center", fontsize=6.5, color=PEC, va="top")

    # --- c: column pointer ---
    blk(ax, 8.55, 5.85, 3.15, 1.20, "col j  sequencer\nISS: raddr = j\nEX:  compute + we",
        fc=WHITE, ec=SEC, fs=7.5)
    blk(ax, 8.55, 4.25, 3.15, 1.20, "v[j],  δ scalar\nδ = β (v[j] − kv) ≫ 8\nbroadcast to all lanes",
        fc=WHITE, ec=SEC, fs=7.2)
    blk(ax, 8.55, 2.55, 3.15, 1.25, "1 issue / cycle\nD columns + 2 pipe\n= 130 clk  (measured)",
        fc=WHITE, ec=SEC, fs=7.2)
    arr(ax, 10.12, 5.85, 10.12, 5.45, c=SEC)
    arr(ax, 10.12, 4.25, 10.12, 3.80, c=SEC)
    wire(ax, 8.55, 4.85, 7.95, 4.85, c=SEC, lw=1.1)
    arr(ax, 7.95, 4.85, 7.95, 3.42, c=SEC)

    # --- d: trees + write ---
    blk(ax, 12.25, 5.85, 3.55, 1.25, "add_tree_bal  D=128\nkv = (Σ p_k) ≫ 8",
        fc=WHITE, ec=AEC, fs=7.5)
    blk(ax, 12.25, 4.15, 3.55, 1.25, "add_tree_bal  D=128\no[j] = (Σ p_q) ≫ 8",
        fc=WHITE, ec=AEC, fs=7.5)
    blk(ax, 12.25, 2.45, 3.55, 1.25, "write S'[:, j]\npack o[j]  into o_flat",
        fc=WHITE, ec=AEC, fs=7.5)
    arr(ax, 7.95, 4.62, 12.25, 6.47, c=PEC, lw=1.0)
    arr(ax, 7.95, 2.22, 12.25, 4.77, c=PEC, lw=1.0)
    arr(ax, 14.02, 4.15, 14.02, 3.70, c=AEC)
    # writeback to SRAM
    wire(ax, 12.25, 3.07, 3.50, 3.07, c=WEC, lw=1.2)
    arr(ax, 3.50, 3.07, 3.50, 2.55, c=WEC)
    ax.text(7.8, 3.22, "S' writeback", fontsize=6.5, color=WEC, ha="center")

    ax.text(14.02, 1.45, "area here is PEs + S SRAM\nnot 17B vias\n~77 mm² S + ≪10 mm² logic",
            ha="center", fontsize=6.5, color=AEC, va="top")

    ax.text(0.15, 0.08,
            "s_dec = (S[:,j]·g)≫8     kv = (s_dec·k)≫8     δ = β(v[j]−kv)≫8     "
            "S' = sat(s_dec + (k·δ)≫8)     o[j] = (S'·q)≫8     fused, 1 pass, rtl/gated_delta_fused.v",
            fontsize=6.8, color=MUTED, va="bottom")


def main() -> None:
    fig = plt.figure(figsize=(16.2, 13.4), facecolor=WHITE)
    fig.text(0.04, 0.975,
             "Inside the two arrays   ·   Qwen3.8-27B Q4   ·   28 nm",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.948,
             "Same four stages as a via-programmed mat-vec (weight → pre-multiply → select → accumulate).  "
             "FFN stores 17.1B weights as vias.  DeltaNet stores a D×D state SRAM and a handful of vectors.",
             fontsize=8.2, color=MUTED, va="top")

    ax1 = fig.add_axes([0.03, 0.50, 0.95, 0.42])
    draw_ffn(ax1)
    ax2 = fig.add_axes([0.03, 0.03, 0.95, 0.42])
    draw_delta(ax2)

    out = ART / "ffn-via-tap-and-deltanet-inside.png"
    fig.savefig(out, dpi=170, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
