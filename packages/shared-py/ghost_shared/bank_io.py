"""The single (de)serialization implementation for KV banks. CONTRACTS.md §4.

A bank = one page type = K and V tensors for each selected layer.

Per layer L in SELECTED_LAYERS, two arrays each shaped
``[NUM_KV_HEADS=8, num_slots, HEAD_DIM=128]``, dtype float32, C-order.

Keys are canonical *pre-RoPE* (paper Eq. 6-7). num_slots is identical across
the 4 layers within one bank.

B1 (bank-compiler) writes with this module; A1 (inference-engine) and A2's
upload scripts read with it. Do NOT hand-roll a second implementation.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any

import numpy as np

from .constants import BANK_DTYPE, HEAD_DIM, MODEL_ID, NUM_KV_HEADS, SELECTED_LAYERS

_NP_DTYPE = np.dtype(BANK_DTYPE)  # float32
MANIFEST_NAME = "manifest.json"


class BankFormatError(ValueError):
    """Raised when a bank array violates the binary contract."""


# --------------------------------------------------------------------------
# Low-level byte (de)serialization
# --------------------------------------------------------------------------
def to_bytes(arr: np.ndarray) -> bytes:
    """Serialize a single K or V array to raw C-order float32 bytes.

    Raises BankFormatError on wrong dtype, rank, or head/dim sizes.
    """
    if not isinstance(arr, np.ndarray):
        raise BankFormatError(f"expected np.ndarray, got {type(arr)!r}")
    if arr.dtype != _NP_DTYPE:
        raise BankFormatError(f"expected dtype {_NP_DTYPE}, got {arr.dtype}")
    if arr.ndim != 3:
        raise BankFormatError(f"expected 3-D array, got shape {arr.shape}")
    n_heads, num_slots, head_dim = arr.shape
    if n_heads != NUM_KV_HEADS:
        raise BankFormatError(f"expected {NUM_KV_HEADS} KV heads, got {n_heads}")
    if head_dim != HEAD_DIM:
        raise BankFormatError(f"expected head_dim {HEAD_DIM}, got {head_dim}")
    if num_slots < 1:
        raise BankFormatError(f"num_slots must be >= 1, got {num_slots}")
    # np.ascontiguousarray guarantees C-order without copying when already C.
    return np.ascontiguousarray(arr).tobytes()


def from_bytes(buf: bytes, num_slots: int) -> np.ndarray:
    """Deserialize raw bytes back to a ``[8, num_slots, 128]`` float32 array."""
    expected = NUM_KV_HEADS * num_slots * HEAD_DIM * _NP_DTYPE.itemsize
    if len(buf) != expected:
        raise BankFormatError(
            f"byte length {len(buf)} != expected {expected} for num_slots={num_slots}"
        )
    arr = np.frombuffer(buf, dtype=_NP_DTYPE).reshape(NUM_KV_HEADS, num_slots, HEAD_DIM)
    return (
        arr.copy()
    )  # frombuffer is read-only; callers (torch.from_numpy) need writable


# --------------------------------------------------------------------------
# File naming
# --------------------------------------------------------------------------
def bank_filename(page_key: str, layer: int, kind: str) -> str:
    """e.g. page_key='hn:front', layer=8, kind='k' -> 'hn_front__L8__k.bin'."""
    if kind not in ("k", "v"):
        raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")
    safe = page_key.replace(":", "_")
    return f"{safe}__L{layer}__{kind}.bin"


# --------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_manifest(out_dir: str) -> dict[str, Any]:
    """Read manifest.json from out_dir, or return a fresh skeleton."""
    path = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "model_id": MODEL_ID,
        "selected_layers": list(SELECTED_LAYERS),
        "banks": [],
    }


def write_manifest(out_dir: str, manifest: dict[str, Any]) -> str:
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


# --------------------------------------------------------------------------
# High-level save / load
# --------------------------------------------------------------------------
def save_bank(
    out_dir: str,
    page_key: str,
    banks: dict[int, tuple[np.ndarray, np.ndarray]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write all .bin files for one bank and upsert its manifest entry.

    ``banks`` maps layer_id -> (k_array, v_array), each ``[8, S, 128]`` f32.
    ``meta`` may carry: domain, dom_structural_hash, summary_text_path.
    Returns the manifest entry that was written.
    """
    meta = dict(meta or {})
    layers = sorted(banks.keys())
    if layers != sorted(SELECTED_LAYERS):
        raise BankFormatError(
            f"bank must cover exactly layers {sorted(SELECTED_LAYERS)}, got {layers}"
        )

    os.makedirs(out_dir, exist_ok=True)

    num_slots: int | None = None
    files: dict[str, dict[str, str]] = {}
    for layer in layers:
        k_arr, v_arr = banks[layer]
        k_bytes = to_bytes(k_arr)
        v_bytes = to_bytes(v_arr)
        if k_arr.shape != v_arr.shape:
            raise BankFormatError(
                f"layer {layer}: K shape {k_arr.shape} != V shape {v_arr.shape}"
            )
        slots = k_arr.shape[1]
        if num_slots is None:
            num_slots = slots
        elif slots != num_slots:
            raise BankFormatError(
                f"num_slots differ across layers: {num_slots} vs {slots} "
                f"at layer {layer}"
            )
        k_name = bank_filename(page_key, layer, "k")
        v_name = bank_filename(page_key, layer, "v")
        with open(os.path.join(out_dir, k_name), "wb") as fh:
            fh.write(k_bytes)
        with open(os.path.join(out_dir, v_name), "wb") as fh:
            fh.write(v_bytes)
        files[str(layer)] = {"k": k_name, "v": v_name}

    entry: dict[str, Any] = {
        "page_key": page_key,
        "domain": meta.get("domain", ""),
        "num_slots": int(num_slots or 0),
        "dom_structural_hash": meta.get("dom_structural_hash", ""),
        "summary_text_path": meta.get("summary_text_path", ""),
        "files": files,
        "compiled_at": meta.get("compiled_at") or _utc_now_iso(),
    }

    manifest = read_manifest(out_dir)
    manifest["banks"] = [
        b for b in manifest.get("banks", []) if b.get("page_key") != page_key
    ]
    manifest["banks"].append(entry)
    write_manifest(out_dir, manifest)
    return entry


def load_bank(
    manifest_entry: dict[str, Any], banks_dir: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load one bank's K/V arrays from disk given its manifest entry."""
    num_slots = int(manifest_entry["num_slots"])
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_str, names in manifest_entry["files"].items():
        layer = int(layer_str)
        with open(os.path.join(banks_dir, names["k"]), "rb") as fh:
            k_arr = from_bytes(fh.read(), num_slots)
        with open(os.path.join(banks_dir, names["v"]), "rb") as fh:
            v_arr = from_bytes(fh.read(), num_slots)
        out[layer] = (k_arr, v_arr)
    return out


def load_all_banks_from_dir(
    banks_dir: str,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Load every bank listed in ``banks_dir/manifest.json`` keyed by page_key.

    This is the filesystem fallback A1 uses when ClickHouse is unreachable.
    """
    manifest = read_manifest(banks_dir)
    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for entry in manifest.get("banks", []):
        out[entry["page_key"]] = load_bank(entry, banks_dir)
    return out
