"""Op counts for architecture direction checks (no RTL required).

Decode of one new token:
  softmax attention  ~ 2 * seq * D   (QK + AV; softmax ignored)
  Gated DeltaNet     ~ 4 * D^2 + D   (decay, kᵀS, outer, qᵀS; delta is D)
  FFN mat-vec        ~ M * N         independent of seq

DeltaNet work does not grow with context; attention decode does.
"""

from __future__ import annotations


def deltanet_macs_per_token(d: int) -> int:
    return 4 * d * d + d


def attn_decode_macs(seq: int, d: int) -> int:
    return 2 * seq * d


def ffn_macs(rows: int, cols: int) -> int:
    return rows * cols


def crossover_seq(d: int) -> float:
    """seq where attn decode MACs exceed DeltaNet MACs: 2*seq*D = 4*D^2+D."""
    return (4 * d * d + d) / (2 * d)


def complexity_table(d: int, seqs: tuple[int, ...] = (64, 256, 1024, 4096, 32768)) -> list[dict]:
    delta = deltanet_macs_per_token(d)
    rows = []
    for seq in seqs:
        attn = attn_decode_macs(seq, d)
        rows.append(
            {
                "seq": seq,
                "d": d,
                "delta_macs": delta,
                "attn_decode_macs": attn,
                "ratio_attn_over_delta": round(attn / delta, 3) if delta else None,
            }
        )
    return rows
