"""ghost_shared — the single source of shared contract code (CONTRACTS.md).

Exposes page_key(), bank (de)serialization, the structural DOM hash, project
constants, and the ClickHouse storage client. Every Python app imports from
here; there must be exactly one implementation of each.
"""

from __future__ import annotations

from . import bank_io, constants, dom_hash, page_key, storage
from .bank_io import (
    BankFormatError,
    bank_filename,
    from_bytes,
    load_all_banks_from_dir,
    load_bank,
    read_manifest,
    save_bank,
    to_bytes,
    write_manifest,
)
from .constants import (
    BANK_DTYPE,
    HAIKU_MODEL,
    HEAD_DIM,
    HIDDEN_SIZE,
    MODEL_ID,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    SELECTED_LAYERS,
    SUMMARY_WORDS,
)
from .dom_hash import dom_structural_hash
from .page_key import page_key as compute_page_key

__all__ = [
    "bank_io",
    "constants",
    "dom_hash",
    "page_key",
    "storage",
    "compute_page_key",
    "dom_structural_hash",
    "save_bank",
    "load_bank",
    "load_all_banks_from_dir",
    "to_bytes",
    "from_bytes",
    "bank_filename",
    "read_manifest",
    "write_manifest",
    "BankFormatError",
    "MODEL_ID",
    "SELECTED_LAYERS",
    "NUM_LAYERS",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "BANK_DTYPE",
    "SUMMARY_WORDS",
    "HAIKU_MODEL",
]

__version__ = "0.1.0"
