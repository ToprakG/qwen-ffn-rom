#!/usr/bin/env python3
"""Qwen3.8-27B Q4 28 nm area: FFN via-ROM vs DeltaNet+rest, to scale.

  python scripts/draw_area_q4_28nm.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

INK = "#1a1a1a"
MUTED = "#5c5c5c"
RULE = "#d0d0d0"
MIXER = "#2c5f8a"
ATTN = "#c23b22"
FFN = "#c9a227"
OK = "#2d7a4f"
WHITE = "#ffffff"
FILL = "#f6f6f4"
SRAM = "#5a7a62"

# Same estimator as rom_estimator.py / eda_sky130_blocks.py
RETICLE_W, RETICLE_H = 26.0, 33.0
RETICLE = RETICLE_W * RETICLE_H  # 858 mm²
CELL = 0.015875  # µm², max(18 F², SRAM/8) at 28 nm
EFF = 0.62
BITS = 4
H, I, L = 5120, 17408, 64
VOCAB = 248320
N_Q, N_KV, D_ATTN = 24, 4, 256
DELTA_L, ATTN_L = 48, 16
SRAM_UM2 = 0.127
V_HEADS, D = 48, 128


def rom_mm2(n: float) -> float:
    return n * BITS * CELL / EFF / 1e6


def sram_mm2(n_bits: float) -> float:
    return n_bits * SRAM_UM2 / 1e6


def box(ax, x, y, w, h, fc, title, body, *, ec=INK, lw=1.0, title_c=INK, fs=8.0, bfs=7.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.35,
    )
    ax.add_patch(p)
    ax.text(x + 0.08, y + h - 0.10, title, fontsize=fs, fontweight="bold",
            color=title_c, va="top")
    ax.text(x + 0.08, y + h - 0.34, body, fontsize=bfs, color=INK,
            va="top", linespacing=1.38)


def panel_scale(ax, area_mm2: float, color: str, label: str) -> None:
    """Plan view in millimetres: reticle vs a square of `area_mm2`."""
    ax.set_xlim(-2, 56)
    ax.set_ylim(-4, 50)
    ax.set_aspect("equal")
    ax.axis("off")

    side = area_mm2 ** 0.5
    # Reticle lower-left at origin
    ax.add_patch(Rectangle((0, 0), RETICLE_W, RETICLE_H, fill=False,
                           edgecolor=OK, linewidth=2.0, zorder=3))
    ax.add_patch(Rectangle((0, 0), RETICLE_W, RETICLE_H,
                           facecolor=OK, alpha=0.06, zorder=0))
    ax.text(RETICLE_W / 2, RETICLE_H / 2, "one reticle\n26 × 33 mm\n858 mm²",
            ha="center", va="center", fontsize=8, color=OK, fontweight="bold",
            zorder=4, linespacing=1.4)

    # Array square, origin-aligned so overflow is obvious
    ax.add_patch(Rectangle((0, 0), side, side, fill=True,
                           facecolor=color, alpha=0.28, edgecolor=color,
                           linewidth=1.8, zorder=2))
    ax.annotate(
        f"{label}\n{side:.1f} × {side:.1f} mm\n{area_mm2:,.0f} mm²",
        xy=(side, side), xytext=(side + 3.2, side - 2),
        fontsize=8, color=color, fontweight="bold", va="top",
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
    )
    ax.text(0, -2.6, "same origin · millimetres · square of equal area",
            fontsize=7, color=MUTED, va="top")


def main() -> None:
    ffn_w = 3 * H * I * L
    embed_w = VOCAB * H
    attn_w = ATTN_L * (
        H * (N_Q * D_ATTN) + 2 * H * (N_KV * D_ATTN) + (N_Q * D_ATTN) * H
    )
    rest_w = 27e9 - ffn_w  # non-FFN so the 27B rounding is exact
    delta_w = rest_w - embed_w - attn_w

    ffn_a = rom_mm2(ffn_w)
    embed_a = rom_mm2(embed_w)
    attn_a = rom_mm2(attn_w)
    delta_a = rom_mm2(delta_w)
    rest_a = rom_mm2(rest_w)

    s_bits = DELTA_L * V_HEADS * D * D * 16
    kv4k = ATTN_L * 2 * N_KV * D_ATTN * 4096 * 4
    kv32k = ATTN_L * 2 * N_KV * D_ATTN * 32768 * 4
    s_a = sram_mm2(s_bits)
    kv4_a = sram_mm2(kv4k)
    kv32_a = sram_mm2(kv32k)

    fig = plt.figure(figsize=(16.0, 10.2), facecolor=WHITE)
    fig.text(0.045, 0.965,
             "Qwen3.8-27B at Q4, 28 nm — FFN via-ROM vs DeltaNet + rest",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.928,
             "Die area = params × 4 bit × 0.015875 µm² / 0.62.  "
             "Cell = max(18 F², SRAM/8).  Reticle = 26 × 33 mm = 858 mm².  "
             "Compute PEs are ≪ 10 mm² and are not on this map.",
             fontsize=8.2, color=MUTED, va="top")

    # -------- LEFT: FFN --------
    ax = fig.add_axes([0.04, 0.42, 0.44, 0.46])
    panel_scale(ax, ffn_a, FFN, "FFN array")

    axf = fig.add_axes([0.04, 0.06, 0.44, 0.34])
    axf.set_xlim(0, 10)
    axf.set_ylim(0, 6.2)
    axf.axis("off")
    axf.set_title("FFN  ·  via-ROM  ·  every layer", loc="left",
                  fontsize=11, fontweight="bold", color=FFN, pad=2)

    box(axf, 0.05, 4.15, 9.7, 1.85, FILL,
        "64 × SwiGLU   gate + up + down",
        f"per layer  3 × {H} × {I:,}  =  {3*H*I/1e6:.1f} M weights\n"
        f"64 layers                   =  {ffn_w/1e9:.3f} B weights   =  {ffn_w*4/8/1e9:.2f} GB at Q4\n"
        f"mm² = 17.113e9 × 4 × 0.015875 / 0.62 / 1e6  =  {ffn_a:,.0f} mm²",
        ec=FFN, title_c=FFN, lw=1.3)

    # three matrix tiles
    names = [
        ("gate  I×H", "17408 × 5120"),
        ("up    I×H", "17408 × 5120"),
        ("down  H×I", "5120 × 17408"),
    ]
    for i, (t, s) in enumerate(names):
        x = 0.05 + i * 3.25
        box(axf, x, 2.15, 3.1, 1.75, WHITE, t, s + "\nvia tap: ±x, x≪1, x≪2, −x≪3\nnot a stdcell MAC",
            ec=FFN, fs=7.6, bfs=6.6)

    axf.text(0.08, 1.85, "Why it misses one 28 nm shot", fontsize=8.5,
             fontweight="bold", color=INK, va="top")
    axf.text(0.08, 1.52,
             f"{ffn_a:,.0f} / 858  =  {ffn_a/RETICLE:.2f}× reticle.  "
             f"A square of that area is {ffn_a**0.5:.1f} mm on a side; the field is 26 × 33 mm.\n"
             "Even uncapped 18 F² (0.01411 µm²) is 1,558 mm² — still 1.82×.  "
             "2:4 or ~2 bit would be required to fit FFN alone.\n"
             "Stdcell CSD (~500 µm²/weight) is not added: the farm is the ROM.",
             fontsize=7.3, color=INK, va="top", linespacing=1.45)

    # -------- RIGHT: DeltaNet + rest --------
    axr = fig.add_axes([0.52, 0.42, 0.46, 0.46])
    panel_scale(axr, rest_a, MIXER, "non-FFN ROM")

    axd = fig.add_axes([0.52, 0.06, 0.46, 0.34])
    axd.set_xlim(0, 10)
    axd.set_ylim(0, 6.2)
    axd.axis("off")
    axd.set_title("DeltaNet + rest  ·  not the FFN", loc="left",
                  fontsize=11, fontweight="bold", color=MIXER, pad=2)

    box(axd, 0.05, 4.15, 9.7, 1.85, FILL,
        "27B − FFN  =  9.887 B weights  →  1,013 mm² Q4 ROM",
        "48 Gated DeltaNet layers + 16 Gated Attention + tied embed.\n"
        f"This block alone is {rest_a/RETICLE:.2f}× a reticle — still over, before FFN.",
        ec=MIXER, title_c=MIXER, lw=1.3)

    parts = [
        (MIXER, "DeltaNet + extras", delta_w, delta_a,
         "48 layers · q/k/v/out + gates\nD=128, 16 K / 48 V heads"),
        (ATTN, "Attn QKVO", attn_w, attn_a,
         "16 layers · 24 Q / 4 KV\nhead dim 256"),
        ("#6e6e6e", "Embed (tied)", embed_w, embed_a,
         f"vocab {VOCAB:,} × 5120\nlm_head shares the ROM"),
    ]
    y0 = 2.15
    h0 = 1.75
    xs = [0.05, 3.35, 6.65]
    for x, (c, t, w, a, b) in zip(xs, parts):
        box(axd, x, y0, 3.15, h0, WHITE, t,
            f"{w/1e9:.3f} B   {a:.0f} mm²\n" + b,
            ec=c, title_c=c, fs=7.5, bfs=6.5)

    # stacked bar of rest ROM vs reticle
    axd.add_patch(Rectangle((0.05, 0.22), 9.7, 1.72, facecolor=FILL,
                            edgecolor=RULE, lw=0.8))
    axd.text(0.18, 1.72, "Q4 ROM stack inside non-FFN   vs   SRAM (not vias)",
             fontsize=7.5, fontweight="bold", color=INK, va="top")
    scale = 9.4 / max(rest_a, RETICLE)
    x = 0.18
    for c, a, lab in ((MIXER, delta_a, "Δ"), (ATTN, attn_a, "A"), ("#6e6e6e", embed_a, "E")):
        w = a * scale
        axd.add_patch(Rectangle((x, 1.05), w, 0.38, facecolor=c, edgecolor=c, lw=0))
        x += w
    axd.text(0.18, 0.92, f"non-FFN ROM {rest_a:,.0f} mm²", fontsize=7, color=MUTED, va="top")
    axd.add_patch(Rectangle((0.18, 0.42), RETICLE * scale, 0.28,
                            facecolor="none", edgecolor=OK, lw=1.6))
    axd.text(0.18 + RETICLE * scale + 0.12, 0.55, "858 mm² field",
             fontsize=7, color=OK, va="center", fontweight="bold")

    axd.text(5.55, 1.72, f"S-state int16  {s_a:.0f} mm² SRAM\n"
             f"int4 KV 4k     {kv4_a:.0f} mm²\n"
             f"int4 KV 32k    {kv32_a:.0f} mm²",
             fontsize=6.7, color=SRAM, va="top", fontfamily="sans-serif",
             linespacing=1.45)

    out = ART / "area-q4-28nm-ffn-vs-rest.png"
    fig.savefig(out, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  FFN     {ffn_w/1e9:.3f} B  {ffn_a:.1f} mm2  {ffn_a/RETICLE:.2f}x")
    print(f"  rest    {rest_w/1e9:.3f} B  {rest_a:.1f} mm2  {rest_a/RETICLE:.2f}x")
    print(f"  delta   {delta_w/1e9:.3f} B  {delta_a:.1f} mm2")
    print(f"  attn    {attn_w/1e9:.3f} B  {attn_a:.1f} mm2")
    print(f"  embed   {embed_w/1e9:.3f} B  {embed_a:.1f} mm2")


if __name__ == "__main__":
    main()
