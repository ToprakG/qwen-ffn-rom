"""Load one FFN tensor from a Hugging Face safetensors shard via HTTP range."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import httpx
import numpy as np

HF_URL = (
    "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/"
    "model.safetensors-00001-of-00001.safetensors"
)
TARGET_NAME = "model.language_model.layers.0.mlp.down_proj.weight"


def _get_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "qwen-ffn-rom"}
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


def bf16_to_fp32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).copy()


def load_down_proj(cache_dir: Path, url: str = HF_URL, name: str = TARGET_NAME) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / "down_proj_fp32.npy"
    if npy_path.exists():
        return np.load(npy_path)

    hdr_len = struct.unpack("<Q", _get_range(url, 0, 7))[0]
    header = json.loads(_get_range(url, 8, 7 + hdr_len).decode("utf-8"))
    info = header[name]
    if info["dtype"] != "BF16":
        raise ValueError(f"unexpected dtype {info['dtype']}")
    off0, off1 = info["data_offsets"]
    base = 8 + hdr_len
    raw = _get_range(url, base + off0, base + off1 - 1)
    if len(raw) != off1 - off0:
        raise RuntimeError(f"short read {len(raw)} != {off1 - off0}")
    w = bf16_to_fp32(raw).reshape(info["shape"])
    np.save(npy_path, w)
    return w
