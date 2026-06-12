"""Provenance tagging for compiled banks.

Banks compiled via the real B1/R1 pipeline (DOM → Haiku → Llama → pre-RoPE
K/V) carry ``"synthetic": false`` in manifest.json.  The off-GPU fallback in
``scripts/build_demo_banks.py`` carries ``"synthetic": true``.

This module provides a single helper that toggles the marker on an existing
manifest entry.  The marker is purely additive — old readers that don't know
about the key silently ignore it.
"""

from __future__ import annotations

from ghost_shared import bank_io


def tag_synthetic(
    out_dir: str,
    page_key: str,
    *,
    synthetic: bool,
) -> bool:
    """Set or clear the ``"synthetic"`` flag on a single bank's manifest entry.

    Returns ``True`` if the entry was found and updated, ``False`` if
    *page_key* was not present in the manifest (no-op).
    """
    manifest = bank_io.read_manifest(out_dir)
    found = False
    for entry in manifest.get("banks", []):
        if entry.get("page_key") == page_key:
            entry["synthetic"] = synthetic
            found = True
            break
    if found:
        bank_io.write_manifest(out_dir, manifest)
    return found
