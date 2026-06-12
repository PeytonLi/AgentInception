"""Bank (de)serialization — CONTRACTS.md §4. THE one implementation.

B1 (bank-compiler) writes with it, A1 (inference-engine) / A2 read with it.

Per layer: K and V arrays, each [NUM_KV_HEADS=8, num_slots, HEAD_DIM=128],
float32, C-order, serialized via arr.tobytes(). Keys are canonical pre-RoPE.
num_slots is identical across the selected layers within one bank.
"""

import json
from pathlib import Path

import numpy as np

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SELECTED_LAYERS = [8, 12, 16, 20]
NUM_KV_HEADS = 8
HEAD_DIM = 128
BANK_DTYPE = np.float32

# {page_key: {layer: (k_array, v_array)}}
BankSet = dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]


def serialize_bank_array(arr: np.ndarray) -> bytes:
    if arr.ndim != 3:
        raise ValueError(f"bank array must be [kv_heads, num_slots, head_dim], got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=BANK_DTYPE).tobytes()


def deserialize_bank_array(
    buf: bytes, num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM
) -> np.ndarray:
    slot_bytes = num_kv_heads * head_dim * np.dtype(BANK_DTYPE).itemsize
    if len(buf) == 0 or len(buf) % slot_bytes != 0:
        raise ValueError(
            f"buffer of {len(buf)} bytes is not a whole number of "
            f"[{num_kv_heads}, 1, {head_dim}] float32 slots ({slot_bytes} bytes each)"
        )
    num_slots = len(buf) // slot_bytes
    arr = np.frombuffer(buf, dtype=BANK_DTYPE).reshape(num_kv_heads, num_slots, head_dim)
    return arr.copy()  # frombuffer is read-only; callers (torch.from_numpy) need writable


def bank_file_name(page_key: str, layer: int, kind: str) -> str:
    if kind not in ("k", "v"):
        raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")
    return f"{page_key.replace(':', '_')}__L{layer}__{kind}.bin"


def write_bank_files(
    banks_dir: str | Path,
    page_key: str,
    layer_arrays: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, str]]:
    """Write one bank's .bin files; returns the manifest 'files' mapping."""
    banks_dir = Path(banks_dir)
    banks_dir.mkdir(parents=True, exist_ok=True)

    slot_counts = set()
    for layer, (k, v) in layer_arrays.items():
        if k.shape != v.shape:
            raise ValueError(f"layer {layer}: K shape {k.shape} != V shape {v.shape}")
        slot_counts.add(k.shape[1])
    if len(slot_counts) > 1:
        raise ValueError(f"num_slots must match across layers within one bank, got {sorted(slot_counts)}")

    files: dict[str, dict[str, str]] = {}
    for layer in sorted(layer_arrays):
        k, v = layer_arrays[layer]
        names = {kind: bank_file_name(page_key, layer, kind) for kind in ("k", "v")}
        (banks_dir / names["k"]).write_bytes(serialize_bank_array(k))
        (banks_dir / names["v"]).write_bytes(serialize_bank_array(v))
        files[str(layer)] = names
    return files


def write_manifest(
    banks_dir: str | Path,
    entries: list[dict],
    model_id: str = MODEL_ID,
    selected_layers: list[int] = SELECTED_LAYERS,
) -> Path:
    path = Path(banks_dir) / "manifest.json"
    payload = {"model_id": model_id, "selected_layers": list(selected_layers), "banks": list(entries)}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_manifest(banks_dir: str | Path) -> dict:
    return json.loads((Path(banks_dir) / "manifest.json").read_text())


def load_banks(banks_dir: str | Path) -> BankSet:
    """Read every bank listed in banks_dir/manifest.json into numpy arrays."""
    banks_dir = Path(banks_dir)
    manifest = load_manifest(banks_dir)
    out: BankSet = {}
    for entry in manifest["banks"]:
        per_layer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for layer_str, names in entry["files"].items():
            k = deserialize_bank_array((banks_dir / names["k"]).read_bytes())
            v = deserialize_bank_array((banks_dir / names["v"]).read_bytes())
            per_layer[int(layer_str)] = (k, v)
        out[entry["page_key"]] = per_layer
    return out
