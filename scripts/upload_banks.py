#!/usr/bin/env python
"""Upload compiled banks from a directory into ClickHouse. CONTRACTS.md §5.

Reads <banks_dir>/manifest.json + the referenced .bin files via the shared
bank_io, then inserts each bank with the shared storage client. Idempotent
per page_key (deletes existing rows first).

Usage:
    python scripts/upload_banks.py banks/
    CLICKHOUSE_URL=http://host:8123 python scripts/upload_banks.py banks/

B2 owns the canonical demo upload + validation flow; this is the minimal
A2-provided bridge that also doubles as a storage-path smoke test.
"""

from __future__ import annotations

import sys

from ghost_shared import bank_io, storage


def main(banks_dir: str) -> int:
    manifest = bank_io.read_manifest(banks_dir)
    entries = manifest.get("banks", [])
    if not entries:
        print(f"No banks found in {banks_dir}/manifest.json", file=sys.stderr)
        return 1

    client = storage.get_client()
    for entry in entries:
        page_key = entry["page_key"]
        banks = bank_io.load_bank(entry, banks_dir)
        client.command(
            f"ALTER TABLE {storage.BANKS_TABLE} DELETE WHERE page_key = %(pk)s",
            parameters={"pk": page_key},
        )
        storage.insert_bank(
            client,
            page_key=page_key,
            domain=entry.get("domain", ""),
            banks=banks,
            dom_structural_hash=entry.get("dom_structural_hash", ""),
        )
        print(f"  uploaded {page_key}  ({len(banks)} layers, "
              f"{entry['num_slots']} slots)")

    loaded = storage.load_all_banks(client)
    print(f"ClickHouse now holds banks for: {sorted(loaded.keys())}")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "banks"
    raise SystemExit(main(target))
