"""Canonical signed-digit recoding and Verilog shift-add emission."""

from __future__ import annotations


def to_csd(n: int) -> list[int]:
    """CSD digits low-bit first, each in {-1, 0, +1}, no adjacent nonzeros."""
    if n < 0:
        return [-d for d in to_csd(-n)]
    digits: list[int] = []
    x = int(n)
    while x:
        if x & 1:
            d = 2 - (x & 3)
            digits.append(d)
            x -= d
        else:
            digits.append(0)
        x //= 2
    return digits


def csd_terms(n: int) -> list[tuple[int, int]]:
    return [(i, s) for i, s in enumerate(to_csd(n)) if s]


def eval_csd(n: int, x: int) -> int:
    acc = 0
    for sh, s in csd_terms(n):
        acc += s * (x << sh)
    return acc


def verilog_csd_expr(n: int, var: str = "xs") -> str:
    terms = csd_terms(n)
    if not terms:
        return "0"
    parts: list[str] = []
    for i, (sh, s) in enumerate(terms):
        tok = var if sh == 0 else f"({var} <<< {sh})"
        if i == 0:
            parts.append(tok if s > 0 else f"-{tok}")
        else:
            parts.append((" + " if s > 0 else " - ") + tok)
    return "".join(parts)


def signed_range(bits: int) -> range:
    qmax = (1 << (bits - 1)) - 1
    return range(-qmax, qmax + 1)
