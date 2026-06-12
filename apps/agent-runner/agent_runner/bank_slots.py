"""Resolve a bank's slot count (``S_bank``) from ``banks/manifest.json``.

CONTRACTS s4 makes ``num_slots`` a fixed, compiled property of each bank and
records it in the manifest. The structural KV-cache ratio (README headline
formula) needs that ``S_bank``. The real engine may echo ``num_slots`` in its
step response; when it does not, the runner falls back to this manifest map so
the printed ratio is sourced from real, auditable compile-time metadata rather
than a guessed constant.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("agent_runner.bank_slots")


def _candidate_paths(explicit: str | None) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    env = os.environ.get("BANKS_MANIFEST")
    if env:
        paths.append(Path(env))
    # Walk up from CWD looking for a repo-rooted banks/manifest.json.
    here = Path.cwd()
    for base in (here, *here.parents):
        paths.append(base / "banks" / "manifest.json")
    return paths


def load_num_slots_by_page(manifest_path: str | None = None) -> dict[str, int]:
    """Return ``{page_key: num_slots}`` from the first manifest found.

    Missing/unreadable manifest -> empty map (the runner simply reports a 0
    structural ratio, never crashes).
    """
    for path in _candidate_paths(manifest_path):
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            mapping = {
                bank["page_key"]: int(bank["num_slots"])
                for bank in data.get("banks", [])
                if "page_key" in bank and "num_slots" in bank
            }
            if mapping:
                logger.debug("loaded num_slots from %s: %s", path, mapping)
                return mapping
        except Exception as exc:  # pragma: no cover - depends on fs state
            logger.debug("skipping manifest %s: %s", path, exc)
    return {}
