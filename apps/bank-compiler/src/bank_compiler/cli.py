"""CLI: `python -m bank_compiler compile|validate`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agentinception_shared import bank_io
from agentinception_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS

from .compiler import CompileOptions, run_compile


# --------------------------------------------------------------------------
# `compile`
# --------------------------------------------------------------------------
def _cmd_compile(args: argparse.Namespace) -> int:
    opts = CompileOptions(
        url=args.url,
        html=args.html,
        page_key=args.page_key,
        out_dir=args.out,
        selected_layers=args.layers or list(SELECTED_LAYERS),
    )
    entry = run_compile(opts)
    print(
        f"[bank-compiler] wrote bank for page_key={entry['page_key']!r} "
        f"num_slots={entry['num_slots']} layers={sorted(entry['files'].keys())}"
    )
    print(f"  manifest: {Path(args.out) / bank_io.MANIFEST_NAME}")
    print(f"  summary : {entry.get('summary_text_path')}")
    return 0


# --------------------------------------------------------------------------
# `validate`
# --------------------------------------------------------------------------
def validate_dir(banks_dir: str) -> int:
    """Check manifest consistency, file byte lengths, and dtype/shape via bank_io."""
    manifest_path = Path(banks_dir) / bank_io.MANIFEST_NAME
    if not manifest_path.exists():
        print(f"[validate] FAIL: manifest not found at {manifest_path}")
        return 2

    manifest = bank_io.read_manifest(banks_dir)
    banks = manifest.get("banks", [])
    if not banks:
        print(f"[validate] FAIL: manifest has zero banks ({manifest_path})")
        return 2

    expected_layers = set(manifest.get("selected_layers", list(SELECTED_LAYERS)))
    errors: list[str] = []

    for entry in banks:
        pk = entry.get("page_key", "<missing>")
        num_slots = int(entry.get("num_slots", 0))
        if num_slots < 1:
            errors.append(f"{pk}: num_slots must be >= 1, got {num_slots}")
            continue
        files = entry.get("files") or {}
        layer_ids = {int(k) for k in files.keys()}
        if layer_ids != expected_layers:
            errors.append(
                f"{pk}: layers {sorted(layer_ids)} != expected {sorted(expected_layers)}"
            )
            continue

        expected_bytes = NUM_KV_HEADS * num_slots * HEAD_DIM * 4
        # Per-file byte-length + actual array round-trip via shared-py.
        for layer_str, names in files.items():
            for kind in ("k", "v"):
                fname = names.get(kind)
                if not fname:
                    errors.append(f"{pk}: layer {layer_str} missing {kind!r} file")
                    continue
                path = Path(banks_dir) / fname
                if not path.exists():
                    errors.append(f"{pk}: file not found {path}")
                    continue
                size = path.stat().st_size
                if size != expected_bytes:
                    errors.append(
                        f"{pk}: {fname} byte length {size} != expected {expected_bytes}"
                    )
        # Full shape/dtype sanity via load_bank (raises BankFormatError on mismatch).
        try:
            loaded = bank_io.load_bank(entry, banks_dir)
            for L, (k, v) in loaded.items():
                if k.shape != (NUM_KV_HEADS, num_slots, HEAD_DIM):
                    errors.append(f"{pk} L{L}: bad K shape {k.shape}")
                if v.shape != (NUM_KV_HEADS, num_slots, HEAD_DIM):
                    errors.append(f"{pk} L{L}: bad V shape {v.shape}")
        except Exception as exc:  # noqa: BLE001 — surface load errors
            errors.append(f"{pk}: load failed: {exc}")

    if errors:
        print(f"[validate] FAIL ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1

    # Provenance: warn (don't fail) if any bank is synthetic. Banks compiled
    # via the real B1/R1 pipeline carry `"synthetic": false`; the off-GPU
    # fallback in `scripts/build_demo_banks.py` carries `"synthetic": true`.
    # Older manifests with no key at all are treated as "unknown" (silent).
    synthetic_keys = [b.get("page_key") for b in banks if b.get("synthetic") is True]
    if synthetic_keys:
        print(
            "[validate] WARNING: the following banks are tagged "
            f"synthetic (shape-only noise, NOT real): {synthetic_keys}. "
            "Regenerate with scripts/compile_real_banks.py on a GPU box "
            "before relying on them in a demo."
        )

    print(
        f"[validate] OK — {len(banks)} bank(s), {len(expected_layers)} layer(s) each, manifest at {manifest_path}"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    return validate_dir(args.dir)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bank_compiler",
        description="Offline KV-bank compiler (B1). See docs/handoff/rahul/01-bank-compiler.md",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="Compile one page into a KV bank")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Public URL to load via Playwright")
    src.add_argument("--html", help="Local HTML file path")
    c.add_argument(
        "--page-key",
        required=True,
        help="One of hn:front | hn:item | popup:demo | unknown",
    )
    c.add_argument("--out", default="banks/", help="Output directory (default: banks/)")
    c.add_argument(
        "--layers",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="Override SELECTED_LAYERS (comma-separated, e.g. 8,12,16,20)",
    )
    c.set_defaults(func=_cmd_compile)

    v = sub.add_parser("validate", help="Validate a directory of compiled banks")
    v.add_argument("dir", help="banks/ directory containing manifest.json + .bin files")
    v.set_defaults(func=_cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
