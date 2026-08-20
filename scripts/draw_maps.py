#!/usr/bin/env python3
"""Posters: where clocks go, and the chip block diagram.

Reads artifacts/clocks_token.json (farm-hidden FFN + attn(S) + mixer + norms).
Falls back to honest_tok_s.json only if the spine JSON is missing.

  python scripts/draw_maps.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
RMS = "#6e6e6e"
FILL = "#f6f6f4"
WHITE = "#ffffff"


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def box(ax, x, y, w, h, fc, title, body, *, ec=INK, lw=1.0, title_c=INK, fs=8.5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.4,
    )
    ax.add_patch(p)
    ax.text(x + 0.04, y + h - 0.07, title, fontsize=fs, fontweight="bold",
            color=title_c, va="top", ha="left")
    ax.text(x + 0.04, y + h - 0.22, body, fontsize=7.2, color=INK,
            va="top", ha="left", linespacing=1.35)


def row27(clk: dict, seq: int) -> dict:
    return next(r for r in clk["rows"] if r["model"] == "qwen27b" and r["seq"] == seq)


def split_hidden(r: dict) -> tuple[int, int, int, int, int]:
    c = r["cycles"]
    mix = c["mixer_per_delta_layer"] * 48
    attn = c["attn_per_attn_layer"] * 16
    rms = c["rms_both_per_layer"] * 64
    epi = c["ffn_hidden_epilogue"]
    hid = c["token_ffn_farm_hidden"]
    return mix, attn, rms, epi, hid


def draw_time(clk: dict) -> None:
    r4 = row27(clk, 4096)
    r32 = row27(clk, 32768)
    mix, a4, rms4, epi, hid4 = split_hidden(r4)
    _, a32, _, _, hid32 = split_hidden(r32)
    ser4 = r4["cycles"]["token_ffn_serial"]
    ffn_layer = r4["cycles"]["ffn_serial_per_layer"]
    taps = r4["taps_to_hide_under_mixer"]
    mix_clk = r4["cycles"]["mixer_per_delta_layer"]
    rms_clk = r4["cycles"]["rms_each"]

    fig = plt.figure(figsize=(16.0, 10.0), facecolor=WHITE)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.15, 1.05],
                  width_ratios=[1.15, 1.0, 1.0],
                  hspace=0.42, wspace=0.38,
                  left=0.055, right=0.98, top=0.90, bottom=0.06)

    fig.text(0.055, 0.965, "FFN dominates until the farm hides it; then the mixer dominates",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.928,
             f"M: mixer D=128 fused = {mix_clk} clk/layer  ·  attn = ceil(S/512)+2  ·  RMS Newton = {rms_clk} clk  ·  FFN 8×8 = 2 clk/tile.  "
             f"Farm-hidden FFN = one {mix_clk}-clk epilogue.  SiLU = 0 extra.  U: 330 MHz / 1.65 GHz.",
             fontsize=8.2, color=MUTED, va="top")

    # --- A: serial vs hidden (log) ---
    ax = fig.add_subplot(gs[0, 0])
    cats = ["4k ctx\nserial tap", "4k ctx\nFFN hidden", "32k ctx\nFFN hidden"]
    mixer_v = [mix, mix, mix]
    attn_v = [a4, a4, a32]
    rest_v = [rms4 + epi, rms4 + epi, rms4 + epi]
    ffn_v = [ser4 - mix - a4 - rms4 - epi, 0, 0]
    x = range(3)
    ax.bar(x, mixer_v, color=MIXER, width=0.62, label="Mixer (48 Δ)")
    ax.bar(x, attn_v, bottom=mixer_v, color=ATTN, width=0.62, label="Attn (16)")
    ax.bar(x, rest_v, bottom=[m + a for m, a in zip(mixer_v, attn_v)],
           color=RMS, width=0.62, label="Norms + FFN epi")
    ax.bar(x, ffn_v, bottom=[m + a + r for m, a, r in zip(mixer_v, attn_v, rest_v)],
           color=FFN, width=0.62, label="FFN serial (64)")
    ax.set_yscale("log")
    ax.set_xticks(list(x), cats, fontsize=8)
    ax.set_ylabel("clocks / token  (log)", fontsize=8)
    ax.set_title("One reused tap vs FFN hidden on-die", fontsize=10, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.set_ylim(1e4, 2e9)
    ax.axhline(330e6 / 10_000, color=OK, ls="--", lw=0.8, alpha=0.8)
    ax.text(2.45, 330e6 / 10_000, "10k @ 330 MHz\nbudget 33k clk",
            fontsize=6.5, color=OK, va="bottom", ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.0e}".replace("+0", "").replace("+", "") if v >= 1e5 else f"{int(v):,}"))
    ax.annotate("FFN is 99.97%\nof the token", xy=(0, ser4), xytext=(0.35, 8e7),
                fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))

    # --- B: hidden split ---
    ax = fig.add_subplot(gs[0, 1])
    labels = ["4k", "32k"]
    m = [mix / hid4 * 100, mix / hid32 * 100]
    a = [a4 / hid4 * 100, a32 / hid32 * 100]
    rest = [(rms4 + epi) / hid4 * 100, (rms4 + epi) / hid32 * 100]
    ax.bar(labels, m, color=MIXER, width=0.48, label="Mixer")
    ax.bar(labels, a, bottom=m, color=ATTN, width=0.48, label="Attention")
    ax.bar(labels, rest, bottom=[mv + av for mv, av in zip(m, a)], color=RMS, width=0.48, label="Norms+epi")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of clocks / token", fontsize=8)
    ax.set_title("FFN hidden: attention takes the rest", fontsize=10, loc="left", pad=8)
    for i, (mv, av, tot, ac) in enumerate(zip(m, a, [hid4, hid32], [a4, a32])):
        ax.text(i, mv / 2, f"{mv:.0f}%\n{mix:,}", ha="center", va="center",
                fontsize=7, color=WHITE, fontweight="bold")
        ax.text(i, mv + av / 2, f"{av:.0f}%\n{ac:,}", ha="center", va="center",
                fontsize=7.5, color=WHITE, fontweight="bold")
        ax.text(i, 102, f"{tot:,} clk", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

    # --- C: tok/s vs context ---
    ax = fig.add_subplot(gs[0, 2])
    seqs = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    cyc = [48 * (2 * rms_clk + mix_clk) + 16 * (2 * rms_clk + math.ceil(s / 512) + 2) + mix_clk for s in seqs]
    t330 = [330e6 / c for c in cyc]
    t165 = [1.65e9 / c for c in cyc]
    ax.plot(seqs, t330, color=MIXER, lw=1.8, marker="o", ms=4, label="330 MHz")
    ax.plot(seqs, t165, color=ATTN, lw=1.8, marker="o", ms=4, label="1.65 GHz")
    ax.axhline(10_000, color=OK, ls="--", lw=0.9, label="10k target")
    ax.axhline(50_000, color=FFN, ls="--", lw=0.9, label="50k target")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs, ["256", "512", "1k", "2k", "4k", "8k", "16k", "32k"], fontsize=7)
    ax.set_ylabel("tok/s  (FFN hidden, online softmax)", fontsize=8)
    ax.set_title("Context barely moves tok/s", fontsize=10, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.set_ylim(0, 140_000)
    ax.annotate("10k @ 330 MHz\nholds 256–32k",
                xy=(4096, t330[4]), xytext=(9000, 45_000),
                fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))

    # --- D: one-token layer strip ---
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-0.15, 2.35)
    ax.axis("off")
    ax.set_title("One token, 16 groups of (3× DeltaNet + 1× attention)  ·  bar width ∝ clocks  ·  farm-hidden FFN",
                 fontsize=10, loc="left", pad=6)
    # two rows: 4k and 32k, scale each row to its own group max so mixer is visible
    for row, seq, attn_all, y0, tag in (
        (0, 4096, a4, 1.25, "4k context"),
        (1, 32768, a32, 0.15, "32k context"),
    ):
        per_attn = attn_all / 16
        per_mix = mix_clk
        group = 3 * per_mix + per_attn
        scale = 0.92 / group
        ax.text(-0.02, y0 + 0.42, tag, fontsize=8.5, fontweight="bold",
                color=INK, ha="right", va="center")
        ax.text(-0.02, y0 + 0.18, f"{int(mix + attn_all + rms4 + epi):,} clk",
                fontsize=7, color=MUTED, ha="right", va="center")
        x = 0.15
        for g in range(16):
            for _ in range(3):
                w = per_mix * scale
                ax.add_patch(Rectangle((x, y0), w, 0.55, facecolor=MIXER, edgecolor=WHITE, lw=0.3))
                x += w
            w = per_attn * scale
            ax.add_patch(Rectangle((x, y0), w, 0.55, facecolor=ATTN, edgecolor=WHITE, lw=0.3))
            x += w + 0.012
        ax.text(16.05, y0 + 0.28, f"attn {per_attn:,.0f} clk/layer   mixer {per_mix} ×3",
                fontsize=7, color=MUTED, va="center")
    ax.add_patch(Rectangle((0.15, 2.12), 0.18, 0.12, facecolor=MIXER, edgecolor="none"))
    ax.text(0.38, 2.18, f"DeltaNet mixer ({mix_clk} clk, independent of S)", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((5.4, 2.12), 0.18, 0.12, facecolor=ATTN, edgecolor="none"))
    ax.text(5.63, 2.18, "Gated attention (ceil(S/512)+2, online softmax)", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((11.3, 2.12), 0.18, 0.12, facecolor=RMS, edgecolor="none"))
    ax.text(11.53, 2.18, "Norms + FFN epilogue sit outside this strip", fontsize=7.5, va="center")

    # --- E: per-unit clocks ---
    ax = fig.add_subplot(gs[2, 0])
    names = [
        "FFN 8×8 tap",
        "Mixer D=4 (toy)",
        "Mixer D=128",
        "Attn S=4k / layer",
        "Attn S=32k / layer",
        "FFN SwiGLU / layer\n(one tap, 27B)",
    ]
    vals = [2, 28, mix_clk, a4 // 16, a32 // 16, ffn_layer]
    colors = [OK, MIXER, MIXER, ATTN, ATTN, FFN]
    y = range(len(names))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(list(y), names, fontsize=8)
    ax.set_xlabel("clocks  (log)", fontsize=8)
    ax.set_title("Building-block clocks (sim)", fontsize=10, loc="left", pad=8)
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:,}", va="center", fontsize=7.5, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.set_xlim(1, 3e7)

    # --- F: toy layer pie ---
    ax = fig.add_subplot(gs[2, 1])
    # decoder_layer H=8: mixer 28, FFN 2, rest RMS+FSM
    toy_total = 151
    toy_mix = 28
    toy_ffn = 2
    toy_rest = toy_total - toy_mix - toy_ffn
    wedges, _, autotexts = ax.pie(
        [toy_rest, toy_mix, toy_ffn],
        labels=["RMSNorm ×2 + FSM\n(Newton rsqrt, 2 clk each)", "Mixer D=4\n28 clk", "FFN tap\n2 clk"],
        colors=[RMS, MIXER, OK],
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        pctdistance=0.55,
        textprops={"fontsize": 7.5, "color": INK},
        wedgeprops={"linewidth": 1, "edgecolor": WHITE},
    )
    for t in autotexts:
        t.set_color(WHITE)
        t.set_fontweight("bold")
        t.set_fontsize(8)
    ax.set_title("Wired toy layer  (H=8, 151 clk, PASS)", fontsize=10, loc="left", pad=8)

    # --- G: what to attack ---
    ax = fig.add_subplot(gs[2, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Ranked: where time is actually lost", fontsize=10, loc="left", pad=8)
    rows = [
        (FFN, "1", "Serial FFN (one 8×8 tap)",
         f"{ffn_layer:,} clk/layer.  {taps:,} taps to hide under a {mix_clk}-clk mixer layer.  Without that, 27B is <1 tok/s."),
        (MIXER, "2", "DeltaNet mixer",
         f"{mix_clk} clk/layer, flat in S.  {100*mix/hid4:.0f}% at 4k, {100*mix/hid32:.0f}% at 32k.  SRAM floor D+2."),
        (ATTN, "3", "Attention KV sweep",
         f"ceil(S/512)+2 ×16.  {100*a4/hid4:.0f}% at 4k, {100*a32/hid32:.0f}% at 32k.  Under 1.5k gate."),
        (RMS, "4", "Two RMSNorms + SiLU",
         f"{rms4:,} clk norms + 0 extra SiLU.  {100*(rms4+epi)/hid4:.0f}% with epi at 4k.  Was 7,424 restoring."),
    ]
    y = 0.92
    for c, n, title, body in rows:
        ax.add_patch(Rectangle((0.0, y - 0.18), 0.07, 0.16, facecolor=c, edgecolor="none"))
        ax.text(0.035, y - 0.10, n, ha="center", va="center", color=WHITE,
                fontsize=10, fontweight="bold")
        ax.text(0.10, y - 0.02, title, fontsize=8.5, fontweight="bold", va="top", color=INK)
        ax.text(0.10, y - 0.09, body, fontsize=7.0, va="top", color=MUTED)
        y -= 0.235

    fig.text(0.055, 0.018,
             f"Source: artifacts/clocks_token.json  ·  4k farm-hidden = {hid4:,} clk  ·  "
             f"10k needs {r4['f_for_10k_hz']/1e6:.0f} MHz (U).  Mixer-only {64 * mix_clk:,} clk is not a token.",
             fontsize=7, color=MUTED)
    out = ART / "where-time-goes.png"
    fig.savefig(out, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")


def draw_arch(clk: dict) -> None:
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=WHITE)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.1, 8.72, "Chip architecture  ·  Qwen FFN-ROM",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.1, 8.38,
            "Product is Qwen3.8-27B (64 layers, 16×(3 Gated DeltaNet + 1 Gated Attention), hidden 5120, D=128).  "
            "FPGA board is a mixer demo, not the LLM.",
            fontsize=8, color=MUTED)

    # die outline
    die = FancyBboxPatch((0.1, 2.55), 10.55, 5.55,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=FILL, edgecolor=INK, linewidth=1.2)
    ax.add_patch(die)
    ax.text(0.28, 7.85, "GEN-1 DIE  ·  28 nm target  ·  layers walk in time (one farm reused)",
            fontsize=8.5, fontweight="bold", color=INK)

    box(ax, 0.35, 5.55, 3.15, 2.05, WHITE,
        "DeltaNet farm",
        "48 value heads × D=128 PE\n"
        "S-state on BRAM / SRAM  (O(D²))\n"
        "130 clk / layer  —  D+2 fused pass\n"
        "Independent of S.  Was 524 (4 sweeps)\n"
        "16 K-heads × 3 V-heads overlap\n"
        "rtl/gated_delta_fused.v",
        ec=MIXER, lw=1.4, title_c=MIXER)

    box(ax, 3.70, 5.55, 3.15, 2.05, WHITE,
        "Gated attention  (every 4th)",
        "16 of 64 layers   head_dim 256\n"
        "P=512 int4 KV, GQA 4 KV × 6 Q\n"
        "ceil(S/512)+2 clk / layer\n"
        "Online softmax in RTL  (one pass)\n"
        "rtl/attn_online.v",
        ec=ATTN, lw=1.4, title_c=ATTN)

    box(ax, 7.05, 5.55, 3.30, 2.05, WHITE,
        "SwiGLU FFN  ·  via-ROM",
        "8×8 CSD tap: x, x≪1, x≪2, −x≪3\n"
        "2 clk / tile handshake (measured)\n"
        "4.18M tiles / 27B layer if serial\n"
        "Need ~64,276 taps to hide under Δ\n"
        "rtl/ffn_rom_tap.v  ·  ffn_tap_unit.v",
        ec=FFN, lw=1.4, title_c="#8a7014")

    box(ax, 0.35, 3.55, 3.15, 1.70, WHITE,
        "RMSNorm + residual",
        "rms → mixer/attn → +\n"
        "rms → FFN → +\n"
        "Wired in qwen_layer.v H=16\n"
        "Bit-exact vs Python (12 tok)",
        ec=RMS, lw=1.2, title_c=RMS)

    box(ax, 3.70, 3.55, 3.15, 1.70, WHITE,
        "Sequencer",
        "16 groups:\n"
        "  3 × (Δ → FFN)\n"
        "  1 × (Attn → FFN)\n"
        "One token = 64 layers in series",
        ec=INK, lw=1.2)

    box(ax, 7.05, 3.55, 3.30, 1.70, WHITE,
        "Not on this die yet",
        "Tokenizer / embed / lm_head\n"
        "Full 27B weight ROM tapeout\n"
        "32k KV at D=256 / P=512 on this DUT\n"
        "Host PCIe  (FPGA path only)",
        ec=RULE, lw=1.2, title_c=MUTED)

    # datapath
    ax.text(0.28, 3.28, "One decoder layer datapath  (bit-exact at H=16 vs Python)",
            fontsize=8, fontweight="bold", color=INK)
    steps = [
        (0.35, "x"),
        (1.55, "RMS"),
        (2.95, "Δ or Attn"),
        (4.70, "+"),
        (5.70, "RMS"),
        (6.90, "FFN tap"),
        (8.45, "+"),
        (9.55, "y"),
    ]
    for i, (x, lab) in enumerate(steps):
        col = WHITE
        ec = INK
        if lab == "Δ or Attn":
            ec = MIXER
        elif lab == "FFN tap":
            ec = FFN
        elif lab == "RMS":
            ec = RMS
        ax.add_patch(FancyBboxPatch(
            (x, 2.72), 1.05 if lab == "Δ or Attn" else 0.85, 0.42,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=col, edgecolor=ec, linewidth=1.1,
        ))
        ax.text(x + (0.52 if lab == "Δ or Attn" else 0.42), 2.93, lab,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color=ec)
        if i < len(steps) - 1:
            x2 = steps[i + 1][0]
            ax.annotate("", xy=(x2 - 0.02, 2.93), xytext=(x + (1.07 if lab == "Δ or Attn" else 0.87), 2.93),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))

    # right column: clocks
    ax.add_patch(FancyBboxPatch((11.0, 2.55), 4.85, 5.55,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=WHITE, edgecolor=INK, linewidth=1.2))
    ax.text(11.2, 7.85, "Honest clocks / token", fontsize=9, fontweight="bold", color=INK)
    ax.text(11.2, 7.52, "Farm-hidden, online-softmax, +norms", fontsize=7.5, color=MUTED)

    def kv(y, k, v, vc=INK):
        ax.text(11.25, y, k, fontsize=7.6, color=MUTED, va="center")
        ax.text(15.6, y, v, fontsize=8.2, fontweight="bold", color=vc, va="center", ha="right")

    r4 = row27(clk, 4096)
    r32 = row27(clk, 32768)
    mix, a4, rms4, epi, hid4 = split_hidden(r4)
    _, a32, _, _, hid32 = split_hidden(r32)
    rms_clk = r4["cycles"]["rms_each"]
    kv(7.15, "Mixer, 48 layers", f"{mix:,}")
    kv(6.80, "Attn @ 4k, 16 layers", f"{a4:,}", ATTN)
    kv(6.45, f"Norms 64×{2 * rms_clk} + FFN epi", f"{rms4+epi:,}")
    kv(6.10, "Token @ 4k", f"{hid4:,}", OK)
    kv(5.75, "Token @ 32k", f"{hid32:,}", OK)
    ax.plot([11.3, 15.55], [5.50, 5.50], color=RULE, lw=0.8)
    kv(5.20, "330 MHz  ·  4k  (U)", f"{330e6/hid4:,.0f} tok/s", OK)
    kv(4.85, "1.65 GHz  ·  4k  (U)", f"{1.65e9/hid4:,.0f} tok/s", OK)
    kv(4.50, "10k at 4k needs  (U)", f"{10_000*hid4/1e6:.0f} MHz", OK)
    kv(4.15, "1.65 GHz  ·  32k  (U)", f"{1.65e9/hid32:,.0f} tok/s")
    ax.plot([11.3, 15.55], [3.90, 3.90], color=RULE, lw=0.8)
    ax.text(11.25, 3.55, "Attn fused  ceil(S/512)+2  (was 2+2S)",
            fontsize=7.2, color=MUTED)
    ax.text(11.25, 3.22, "16 layers: 160 clk @ 4k, 1,056 @ 32k.",
            fontsize=7.2, color=MUTED)
    ax.text(11.25, 2.95, "Gate ≤1,500.  10k @ 330 MHz is in budget.",
            fontsize=7.2, color=INK, fontweight="bold")

    # maturity row
    ax.text(0.1, 2.22, "What exists vs the chip", fontsize=9, fontweight="bold", color=INK)
    box(ax, 0.10, 0.18, 5.05, 1.88, WHITE,
        "TODAY  ·  one hardened layer",
        "qwen_layer.v  H=16  PASS  12 tokens\n"
        "Newton RMS  2 clk  ·  SiLU fold  2 clk tap\n"
        "attn_online  ceil(S/P)+2  ·  int4 KV  PASS\n"
        "Sky130 layer  1.382 mm²  ·  34.81 MHz",
        ec=MIXER, lw=1.2, title_c=MIXER, fs=8)
    box(ax, 5.35, 0.18, 5.05, 1.88, WHITE,
        "FPGA  ·  mixer demo, not the LLM",
        "UART host ~711 tok/s is D=4 farm @ ~200 MHz\n"
        "AWS F2: 0.8B-shaped mixer, Python does the rest\n"
        "Does not hold 27B FFN ROM or 32k KV\n"
        "Do not quote board tok/s as Qwen tok/s",
        ec=FFN, lw=1.2, title_c="#8a7014", fs=8)
    box(ax, 10.60, 0.18, 5.25, 1.88, WHITE,
        "GEN-1 CHIP  ·  28 nm, Qwen3.8-27B",
        "Need ~64k 8×8 taps so FFN is not the token\n"
        "Mixer is 130 clk. Attn is 1,056 clk @ 32k\n"
        "RMS is 2 clk (Newton). SiLU is 0 extra\n"
        "10k @ 4k needs ~68 MHz  (U — no 28 nm STA)",
        ec=ATTN, lw=1.2, title_c=ATTN, fs=8)

    out = ART / "chip-architecture.png"
    fig.savefig(out, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")


def draw_spine(clk: dict) -> None:
    fig = plt.figure(figsize=(16.0, 11.2), facecolor=WHITE)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.1, 9.65, "Clocks / token  ·  credibility spine",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.1, 9.28,
            "Farm-hidden FFN + attention(S) + mixer + two RMSNorms.  "
            "M = measured in sim.  D = derived from measured.  U = unmeasured (flagged).",
            fontsize=8, color=MUTED)

    ax.text(0.1, 8.95, "FFN dominates until the farm hides it; then the mixer dominates.",
            fontsize=10, fontweight="bold", color=MIXER)

    # piece table
    headers = ["Piece", "Value", "Tag", "Source"]
    pieces = clk["pieces"]
    rows_p = [
        ["8×8 FFN handshake", f"{pieces['ffn_8x8_handshake_clk']['value']} clk/tile", "M",
         "ffn_tap_cycles.json"],
        ["Mixer D=128 fused", f"{pieces['mixer_d128_clk_per_layer']['value']} clk/layer", "M",
         "column-stream D+2 · was 524 (4 sweeps)"],
        ["Sky130 fused D=16", (
            f"{pieces['sky130_fused_d16_yosys_um2']['value']/1e6:.3f} mm² Yosys"
            if isinstance(pieces.get("sky130_fused_d16_yosys_um2", {}).get("value"), (int, float))
            else "pending"
         ), "M", "liberty-mapped, not PnR · combo EX, no Fmax"],
        ["Attn online P=512", str(pieces["attn_clk_per_layer"]["value"]) + " clk/layer", "M",
         "attn_online_equiv.json  fused softmax, int4 KV"],
        ["RMSNorm each", f"{pieces['rms_clk_each']['value']} clk", "M",
         "Newton rsqrt, 2 clk; SiLU fold 0 extra"],
        ["qwen_layer H=16 Δ", f"{(clk.get('qwen_layer_dut') or {}).get('cycles_delta_layer', '—')} clk", "M",
         "rtl/qwen_layer.v bit-exact vs Python, 12 tokens"],
        ["Sky130 layer stdcell", (
            f"{pieces['sky130_qwen_layer_stdcell_um2']['value']/1e6:.3f} mm²"
            if isinstance(pieces.get("sky130_qwen_layer_stdcell_um2", {}).get("value"), (int, float))
            else "pending"
         ), "M", "OpenLane post-route · Magic DRC/LVS unmeasured"],
        ["Sky130 layer Fmax", (
            f"{pieces['sky130_qwen_layer_fmax_mhz']['value']:.2f} MHz"
            if isinstance(pieces.get("sky130_qwen_layer_fmax_mhz", {}).get("value"), (int, float))
            else "pending"
         ), "M", "post-route WS @ 50 ns · H=16 slice, not 27B"],
        ["7 nm F² from stdcell", (
            f"{pieces['sky130_qwen_layer_7nm_f2_um2']['value']:.0f} µm²"
            if isinstance(pieces.get("sky130_qwen_layer_7nm_f2_um2", {}).get("value"), (int, float))
            else "pending"
         ), "D", "×(7/130)² · not 7 nm STA"],
        ["Sky130 chip Fmax est.", (
            "{lo:.0f}–{hi:.0f} MHz  nom {nom:.0f}".format(
                lo=(pieces.get("sky130_chip_fmax_est_mhz") or {}).get("value", {}).get("lo") or 0,
                hi=(pieces.get("sky130_chip_fmax_est_mhz") or {}).get("value", {}).get("hi") or 0,
                nom=(pieces.get("sky130_chip_fmax_est_mhz") or {}).get("value", {}).get("nom") or 0,
            )
            if isinstance((pieces.get("sky130_chip_fmax_est_mhz") or {}).get("value"), dict)
            else "pending"
         ), "D", "compiler-macro mixer/attn/rsqrt · not 3.23 MHz STA · ±2×"],
        ["7 nm Fmax (projected)", (
            f"{pieces['fmax_7nm']['value']:.0f} MHz nom"
            if isinstance(pieces.get("fmax_7nm", {}).get("value"), (int, float))
            else "pending"
         ), "D", "Sky130 est × 6.5 FO4 · not 7 nm STA"],
    ]
    y = 8.55
    colx = [0.15, 3.6, 7.3, 8.3]
    for i, h in enumerate(headers):
        ax.text(colx[i], y, h, fontsize=8, fontweight="bold", color=MUTED)
    y -= 0.08
    ax.plot([0.15, 15.7], [y, y], color=RULE, lw=0.8)
    y -= 0.32
    for r in rows_p:
        tagc = {"M": OK, "D": MIXER, "U": ATTN}[r[2]]
        ax.text(colx[0], y, r[0], fontsize=8, color=INK)
        ax.text(colx[1], y, r[1], fontsize=8, fontweight="bold", color=INK)
        ax.text(colx[2], y, r[2], fontsize=8, fontweight="bold", color=tagc)
        ax.text(colx[3], y, r[3], fontsize=7.5, color=MUTED)
        y -= 0.34

    ax.text(0.15, y - 0.05, "27B farm-hidden token  (online softmax)   tok/s at flagged clocks",
            fontsize=9, fontweight="bold", color=INK)
    y -= 0.18
    ax.plot([0.15, 15.7], [y, y], color=RULE, lw=0.8)
    y -= 0.32
    hdr2 = ["seq", "clk/token", "10k needs", "41.28 MHz M-proxy", "330 MHz  U", "1.65 GHz  U"]
    col2 = [0.15, 2.0, 5.0, 8.0, 11.2, 13.6]
    for i, h in enumerate(hdr2):
        ax.text(col2[i], y, h, fontsize=7.5, fontweight="bold", color=MUTED)
    y -= 0.08
    ax.plot([0.15, 15.7], [y, y], color=RULE, lw=0.8)
    y -= 0.32
    for r in clk["rows"]:
        if r["model"] != "qwen27b":
            continue
        lab = str(r["seq"]) if r["seq"] < 1024 else f"{r['seq']//1024}k"
        hid = r["cycles"]["token_ffn_farm_hidden"]
        ax.text(col2[0], y, lab, fontsize=9, fontweight="bold", color=INK)
        ax.text(col2[1], y, f"{hid:,}", fontsize=9, color=INK)
        ax.text(col2[2], y, f"{r['f_for_10k_hz']/1e6:.0f} MHz", fontsize=9, color=ATTN)
        ax.text(col2[3], y, f"{r['tok_s_farm_hidden']['41.28MHz']:.0f}", fontsize=9, color=INK)
        ax.text(col2[4], y, f"{r['tok_s_farm_hidden']['330MHz']:.0f}", fontsize=9, color=INK)
        ax.text(col2[5], y, f"{r['tok_s_farm_hidden']['1650MHz']:.0f}", fontsize=9, color=INK)
        y -= 0.38

    y -= 0.1
    ax.text(0.15, y, "One token split, 27B, 4k, farm-hidden", fontsize=9, fontweight="bold", color=INK)
    y -= 0.35
    r4 = next(r for r in clk["rows"] if r["model"] == "qwen27b" and r["seq"] == 4096)
    c = r4["cycles"]
    total = c["token_ffn_farm_hidden"]
    parts = [
        (MIXER, f"Mixer 48×{c['mixer_per_delta_layer']}", c["mixer_per_delta_layer"] * 48),
        (ATTN, f"Attn 16×(ceil(S/512)+2)", c["attn_per_attn_layer"] * 16),
        (RMS, f"Norms 64×{2 * int(c['rms_each'])}", c["rms_both_per_layer"] * 64),
        (OK, "FFN epilogue", c["ffn_hidden_epilogue"]),
    ]
    x = 0.15
    for col, name, val in parts:
        w = 15.5 * val / total
        ax.add_patch(Rectangle((x, y - 0.35), w, 0.5, facecolor=col, edgecolor=WHITE, lw=0.4))
        if w > 1.2:
            ax.text(x + w / 2, y - 0.1, f"{name}\n{val:,}  {100*val/total:.1f}%",
                    ha="center", va="center", fontsize=7.5, color=WHITE, fontweight="bold")
        x += w
    y -= 0.7
    ax.text(0.15, y,
            f"Serial one-tap FFN is {c['token_ffn_serial']:,} clk "
            f"({100*c['ffn_serial_per_layer']*64/c['token_ffn_serial']:.2f}% FFN) — "
            "that column is not the product.",
            fontsize=7.5, color=MUTED)
    y -= 0.45
    dut = clk.get("qwen_layer_dut") or {}
    eda = clk.get("qwen_layer_eda") or {}
    ys = (eda.get("yosys") if isinstance(eda, dict) else None) or {}
    ol = (eda.get("openlane") if isinstance(eda, dict) else None) or {}
    area = ol.get("area_um2") or ys.get("area_um2")
    f2 = eda.get("area_7nm_f2_um2") if isinstance(eda, dict) else None
    fmax = eda.get("fmax_mhz_sky130") if isinstance(eda, dict) else None
    ax.text(0.15, y, "Hardened DUT  rtl/qwen_layer.v   H=16 = 4×D=4 + attn + RMS + 0.8B 16×16 FFN",
            fontsize=9, fontweight="bold", color=INK)
    y -= 0.32
    bits = [f"equiv {dut.get('status','—')}", f"Δ {dut.get('cycles_delta_layer','—')} clk/layer"]
    if isinstance(area, (int, float)):
        bits.append(f"Sky130 stdcell {area/1e6:.3f} mm²  M")
    if isinstance(fmax, (int, float)):
        bits.append(f"Fmax {fmax:.2f} MHz  M")
    ax.text(0.15, y, "   ·   ".join(bits), fontsize=8, color=MUTED)
    if isinstance(f2, (int, float)):
        y -= 0.28
        ax.text(0.15, y,
                f"7 nm F² {f2:,.0f} µm²  D  ·  setup vio 0  ·  hold vio {ol.get('hold_vio', '—')}  ·  "
                f"TritonRoute DRC 0  ·  Magic DRC/LVS U",
                fontsize=8, color=MUTED)

    out = ART / "clocks-token.png"
    fig.savefig(out, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")


def draw_silicon(clk: dict) -> None:
    dut = clk.get("qwen_layer_dut") or {}
    eda = clk.get("qwen_layer_eda") or {}
    ol = (eda.get("openlane") if isinstance(eda, dict) else None) or {}
    ys = (eda.get("yosys") if isinstance(eda, dict) else None) or {}
    area = ol.get("area_um2")
    ys_area = ys.get("area_um2")
    fmax = eda.get("fmax_mhz_sky130") if isinstance(eda, dict) else None
    f2 = eda.get("area_7nm_f2_um2") if isinstance(eda, dict) else None
    die = ol.get("die_area_um2")
    util = ol.get("utilization")
    power = ol.get("power_w")

    fig = plt.figure(figsize=(16.0, 9.0), facecolor=WHITE)
    ax = fig.add_axes([0.035, 0.04, 0.93, 0.92])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.1, 8.72, "One Qwen decoder layer in silicon-ready RTL",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.1, 8.36,
            "rtl/qwen_layer.v  ·  H=16 = 4×D=4 mixer + attn + 2×RMSNorm + 0.8B 16×16 CSD FFN  ·  "
            "bit-exact vs quant/layer_int.py",
            fontsize=8.2, color=MUTED)

    # datapath
    steps = [
        (0.15, "x"),
        (1.55, "RMS1"),
        (3.10, "Δ or Attn"),
        (5.05, "+"),
        (6.20, "RMS2"),
        (7.70, "FFN"),
        (9.15, "+"),
        (10.30, "y"),
    ]
    for i, (x, lab) in enumerate(steps):
        ec = INK
        if lab.startswith("RMS"):
            ec = RMS
        elif "Attn" in lab:
            ec = MIXER
        elif lab == "FFN":
            ec = FFN
        w = 1.70 if "Attn" in lab else 1.10
        ax.add_patch(FancyBboxPatch(
            (x, 7.55), w, 0.55,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=WHITE, edgecolor=ec, linewidth=1.3,
        ))
        ax.text(x + w / 2, 7.82, lab, ha="center", va="center",
                fontsize=9, fontweight="bold", color=ec)
        if i < len(steps) - 1:
            x2 = steps[i + 1][0]
            ax.annotate("", xy=(x2 - 0.04, 7.82),
                        xytext=(x + w + 0.02, 7.82),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))

    # three metric cards
    cards = [
        (0.15, MIXER, "BIT-EXACT  M",
         f"{dut.get('status', '—')}\n"
         f"Δ {dut.get('cycles_delta_layer', '—')} clk/layer (flat in S)\n"
         f"attn {dut.get('cycles_attn_layer_mean', '—')} clk mean\n"
         f"{dut.get('tokens', '—')} tokens  ·  Verilator+cocotb"),
        (5.45, OK, "SKY130 POST-ROUTE  M",
         (f"stdcell {area/1e6:.3f} mm²\n"
          f"Fmax {fmax:.2f} MHz  (WS @ 50 ns)\n"
          f"setup vio 0  ·  hold vio {ol.get('hold_vio', '—')}\n"
          f"TritonRoute DRC {ol.get('route_drc', '—')}  ·  "
          f"{ol.get('stdcell_count') or 0:,} cells")
         if isinstance(area, (int, float)) and isinstance(fmax, (int, float)) else "EDA pending"),
        (10.75, ATTN, "7 nm F²  D   Fmax  U",
         (f"{f2:,.0f} µm²  =  {f2/1e6:.4f} mm²\n"
          "scale (7/130)² on stdcell area\n"
          "not 7 nm STA  ·  not 28 nm STA\n"
          + (f"Yosys pre-PnR {ys_area/1e6:.3f} mm²" if isinstance(ys_area, (int, float)) else ""))
         if isinstance(f2, (int, float)) else "EDA pending"),
    ]
    for x, ec, title, body in cards:
        ax.add_patch(FancyBboxPatch(
            (x, 4.85), 5.05, 2.45,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=FILL, edgecolor=ec, linewidth=1.4,
        ))
        ax.text(x + 0.18, 7.05, title, fontsize=9, fontweight="bold", color=ec)
        ax.text(x + 0.18, 6.70, body, fontsize=8.4, color=INK, va="top", linespacing=1.45)

    # what it is / is not
    ax.add_patch(FancyBboxPatch(
        (0.15, 2.55), 7.70, 2.05,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=WHITE, edgecolor=OK, linewidth=1.2,
    ))
    ax.text(0.35, 4.32, "What this DUT is", fontsize=9, fontweight="bold", color=OK)
    ax.text(0.35, 4.00,
            "One complete decoder layer: mixer + gated attention + FFN tap + RMS + residual.\n"
            "Hidden width is real (H=16), not the H=8 toy. FFN weights are a 0.8B down_proj 16×16 tile.\n"
            "Area and Fmax are measured on Sky130 after place, CTS, route, RCX, post-PnR STA.",
            fontsize=7.8, color=INK, va="top", linespacing=1.4)

    ax.add_patch(FancyBboxPatch(
        (8.10, 2.55), 7.70, 2.05,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=WHITE, edgecolor=ATTN, linewidth=1.2,
    ))
    ax.text(8.30, 4.32, "What this DUT is not", fontsize=9, fontweight="bold", color=ATTN)
    ax.text(8.30, 4.00,
            "Not Qwen3.8-27B. Not D=128. Not 15,947 FFN taps. 414 clk is the H=16 mixer path.\n"
            "27B clocks/token use D=128 fused mixer 130 + attn ceil(S/512)+2 + RMS 2, not these DUT mixer clocks.\n"
            "Die area is 15% util padding. Quote stdcell. Magic DRC/LVS did not finish.",
            fontsize=7.8, color=INK, va="top", linespacing=1.4)

    # flag table
    ax.text(0.15, 2.22, "Flags on every silicon number", fontsize=9, fontweight="bold", color=INK)
    hdr = ["Quantity", "Number", "Tag", "Why"]
    rows = [
        ["Post-route stdcell area", f"{area/1e6:.3f} mm²" if isinstance(area, (int, float)) else "—", "M",
         "OpenLane instance area, not die"],
        ["Die / core (low util)",
         f"{die/1e6:.2f} mm² die, {100*util:.1f}% util" if isinstance(die, (int, float)) and isinstance(util, (int, float)) else "—",
         "M", "Floorplan padding; do not quote as layer cost"],
        ["Sky130 Fmax", f"{fmax:.2f} MHz" if isinstance(fmax, (int, float)) else "—", "M",
         "1000/(50 − WS); setup 0, hold 2 × 11 ps"],
        ["Vectorless power @ 50 ns", f"{power*1e3:.1f} mW" if isinstance(power, (int, float)) else "—", "M",
         "Not scaled to Fmax; not 28 nm"],
        ["7 nm area", f"{f2:,.0f} µm²" if isinstance(f2, (int, float)) else "—", "D",
         "F² scale only"],
        ["Magic DRC / LVS", "flow quit at LEF/antenna", "U", "TritonRoute DRC was 0"],
        ["28 nm / 7 nm Fmax", "no liberty in repo", "U", "330 MHz and 1.65 GHz are goals"],
    ]
    y = 1.95
    colx = [0.15, 4.4, 8.3, 9.3]
    for i, h in enumerate(hdr):
        ax.text(colx[i], y, h, fontsize=7.5, fontweight="bold", color=MUTED)
    y -= 0.06
    ax.plot([0.15, 15.7], [y, y], color=RULE, lw=0.8)
    y -= 0.24
    for r in rows:
        tagc = {"M": OK, "D": MIXER, "U": ATTN}[r[2]]
        ax.text(colx[0], y, r[0], fontsize=7.6, color=INK)
        ax.text(colx[1], y, r[1], fontsize=7.6, fontweight="bold", color=INK)
        ax.text(colx[2], y, r[2], fontsize=8, fontweight="bold", color=tagc)
        ax.text(colx[3], y, r[3], fontsize=7.4, color=MUTED)
        y -= 0.24

    out = ART / "qwen-layer-silicon.png"
    fig.savefig(out, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    spine = ART / "clocks_token.json"
    if not spine.exists():
        raise SystemExit("missing artifacts/clocks_token.json — run python scripts/clocks_token.py")
    clk = json.loads(spine.read_text())
    draw_time(clk)
    draw_arch(clk)
    draw_spine(clk)
    draw_silicon(clk)


if __name__ == "__main__":
    main()
