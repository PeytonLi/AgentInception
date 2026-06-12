"""In-memory bank registry — CONTRACTS §5.

All banks are read ONCE at startup into dict[page_key][layer] -> (K, V)
float32 torch tensors; no DB calls during navigation. ClickHouse is the
primary source; banks/manifest.json + .bin files are the fallback. Which
path was used is logged loudly and exposed as `.source`.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from ghost_shared import bank_io

logger = logging.getLogger("inference_engine.banks")

BANKS_QUERY = (
    "SELECT page_key, layer_id, num_slots, k_bank, v_bank "
    "FROM ghostbrowser.latent_memory_banks"
)

LayerBanks = dict[int, tuple[torch.Tensor, torch.Tensor]]


def _clickhouse_client(url: str):
    # Isolated for test monkeypatching; imported lazily so the engine still
    # starts when clickhouse-connect is absent/unreachable.
    import clickhouse_connect

    return clickhouse_connect.get_client(dsn=url)


class BankRegistry:
    def __init__(self, banks: dict[str, LayerBanks], source: str):
        self._banks = banks
        self.source = source

    @property
    def page_keys(self) -> list[str]:
        return sorted(self._banks.keys())

    def get(self, page_key: str) -> Optional[LayerBanks]:
        return self._banks.get(page_key)

    def num_slots(self, page_key: str) -> int:
        per_layer = self._banks.get(page_key)
        if not per_layer:
            return 0
        first_k, _ = next(iter(per_layer.values()))
        return first_k.shape[1]

    @classmethod
    def from_clickhouse(cls, url: str) -> "BankRegistry":
        client = _clickhouse_client(url)
        # column_formats keeps the raw float32 bytes intact (String columns
        # would otherwise be utf-8 decoded). Verified against a real server
        # in C1's integration suite.
        result = client.query(
            BANKS_QUERY, column_formats={"k_bank": "bytes", "v_bank": "bytes"}
        )
        banks: dict[str, LayerBanks] = {}
        for page_key, layer_id, num_slots, k_buf, v_buf in result.result_rows:
            k = bank_io.from_bytes(bytes(k_buf), num_slots)
            v = bank_io.from_bytes(bytes(v_buf), num_slots)
            if k.shape[1] != num_slots or v.shape[1] != num_slots:
                raise ValueError(
                    f"bank {page_key!r} layer {layer_id}: stored num_slots={num_slots} "
                    f"but buffers deserialize to K{k.shape} / V{v.shape}"
                )
            banks.setdefault(page_key, {})[int(layer_id)] = (
                torch.from_numpy(k),
                torch.from_numpy(v),
            )
        return cls(banks, source="clickhouse")

    @classmethod
    def from_manifest_dir(cls, banks_dir: str | Path) -> "BankRegistry":
        loaded = bank_io.load_all_banks_from_dir(str(banks_dir))
        if not loaded:
            raise FileNotFoundError(f"no banks found in manifest dir {banks_dir}")
        banks: dict[str, LayerBanks] = {
            page_key: {
                layer: (torch.from_numpy(k), torch.from_numpy(v))
                for layer, (k, v) in per_layer.items()
            }
            for page_key, per_layer in loaded.items()
        }
        return cls(banks, source="manifest")

    @classmethod
    def load(cls, clickhouse_url: str, banks_dir: str | Path) -> "BankRegistry":
        try:
            reg = cls.from_clickhouse(clickhouse_url)
            logger.info(
                "==== BANKS LOADED FROM CLICKHOUSE (%s): %s ====",
                clickhouse_url,
                reg.page_keys,
            )
            return reg
        except Exception:
            logger.warning(
                "==== CLICKHOUSE UNREACHABLE at %s — FALLING BACK to manifest dir %s ====",
                clickhouse_url,
                banks_dir,
                exc_info=True,
            )
        try:
            reg = cls.from_manifest_dir(banks_dir)
            logger.warning(
                "==== BANKS LOADED FROM MANIFEST FALLBACK (%s): %s ====",
                banks_dir,
                reg.page_keys,
            )
            return reg
        except Exception:
            logger.error(
                "==== NO BANKS LOADED (ClickHouse AND manifest failed) — "
                "every step will run in plain-prompt fallback ====",
                exc_info=True,
            )
            return cls({}, source="empty")
