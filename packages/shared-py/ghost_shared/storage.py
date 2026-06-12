"""ClickHouse client for banks + agent step logs. CONTRACTS.md §5.

Single implementation, imported by the inference engine (load_all_banks at
startup) and by A2's upload scripts (insert_bank). All KV bytes are stored in
String columns and MUST be read back with the 'bytes' column format so the
raw float32 payload is never utf-8 mangled.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .bank_io import from_bytes as _from_bytes
from .bank_io import to_bytes as _to_bytes

DEFAULT_URL = "http://localhost:8123"
DATABASE = "ghostbrowser"
BANKS_TABLE = f"{DATABASE}.latent_memory_banks"
STEPS_TABLE = f"{DATABASE}.agent_steps"

_MODE_TO_ENUM = {"baseline": 1, "mi": 2}


def get_client(url: str | None = None):
    """Create a clickhouse-connect client from CLICKHOUSE_URL (or override)."""
    import clickhouse_connect  # imported lazily so tests can run without it

    url = url or os.environ.get("CLICKHOUSE_URL", DEFAULT_URL)
    parsed = urlparse(url)
    secure = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (8443 if secure else 8123)
    username = parsed.username or "default"
    password = parsed.password or ""
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure,
    )


def insert_bank(
    client,
    page_key: str,
    domain: str,
    banks: dict[int, tuple[np.ndarray, np.ndarray]],
    dom_structural_hash: str = "",
) -> None:
    """Insert one row per (page_key, layer) with raw float32 K/V bytes."""
    rows: list[list[Any]] = []
    for layer in sorted(banks.keys()):
        k_arr, v_arr = banks[layer]
        num_slots = k_arr.shape[1]
        rows.append(
            [
                page_key,
                domain,
                int(layer),
                int(num_slots),
                _to_bytes(k_arr),
                _to_bytes(v_arr),
                dom_structural_hash,
            ]
        )
    client.insert(
        BANKS_TABLE,
        rows,
        column_names=[
            "page_key",
            "domain",
            "layer_id",
            "num_slots",
            "k_bank",
            "v_bank",
            "dom_structural_hash",
        ],
    )


def load_all_banks(
    client,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Read all banks into the in-memory dict keyed by page_key -> layer.

    Returned arrays are byte-identical to what was inserted.
    """
    result = client.query(
        f"SELECT page_key, layer_id, num_slots, k_bank, v_bank "
        f"FROM {BANKS_TABLE} ORDER BY page_key, layer_id",
        column_formats={"k_bank": "bytes", "v_bank": "bytes"},
    )
    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for page_key, layer_id, num_slots, k_bytes, v_bytes in result.result_rows:
        k_arr = _from_bytes(bytes(k_bytes), int(num_slots))
        v_arr = _from_bytes(bytes(v_bytes), int(num_slots))
        out.setdefault(page_key, {})[int(layer_id)] = (k_arr, v_arr)
    return out


def log_step(
    client,
    *,
    session_id: str,
    step: int,
    mode: str,
    url: str,
    page_key: str,
    action_json: str,
    visible_tokens: int,
    baseline_tokens: int,
    bank_found: bool,
) -> None:
    """Append a row to agent_steps."""
    if mode not in _MODE_TO_ENUM:
        raise ValueError(f"mode must be 'baseline' or 'mi', got {mode!r}")
    client.insert(
        STEPS_TABLE,
        [
            [
                session_id,
                int(step),
                mode,  # clickhouse-connect maps str -> Enum8 by name
                url,
                page_key,
                action_json,
                int(visible_tokens),
                int(baseline_tokens),
                1 if bank_found else 0,
            ]
        ],
        column_names=[
            "session_id",
            "step",
            "mode",
            "url",
            "page_key",
            "action_json",
            "visible_tokens",
            "baseline_tokens",
            "bank_found",
        ],
    )
