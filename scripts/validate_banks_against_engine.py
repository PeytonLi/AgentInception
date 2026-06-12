#!/usr/bin/env python
"""B2 spec, task #4: shape-validate the deployed banks against the engine.

Two checks, both run by default:

  1. Hit the inference engine's GET /healthz and assert all 3 demo page_keys
     appear in ``banks_loaded``.
  2. For each entry in ``banks/manifest.json``, verify on-disk bytes:
       - every k/v file size  == 8 * num_slots * 128 * 4
       - bank_io.load_bank yields ``[8, num_slots, 128]`` float32 arrays
       - num_slots identical across the 4 layers

Exits 0 on full success, 1 on any failure (printed to stderr).

Usage:
    python scripts/validate_banks_against_engine.py
    INFERENCE_URL=http://<ec2-ip>:8000 python scripts/validate_banks_against_engine.py
    python scripts/validate_banks_against_engine.py --skip-engine     # bytes only
    python scripts/validate_banks_against_engine.py --banks-dir banks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))

from ghost_shared import bank_io  # noqa: E402
from ghost_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS  # noqa: E402

REQUIRED_PAGE_KEYS = {"hn:front", "hn:item", "popup:demo"}
F32_BYTES = 4
EXPECTED_BYTES_PER_SLOT = NUM_KV_HEADS * HEAD_DIM * F32_BYTES  # 4096
DEFAULT_INFERENCE_URL = "http://localhost:8000"


def _print_ok(msg: str) -> None:
    print(f"  ok  {msg}")


def _print_err(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def check_engine_healthz(base_url: str) -> list[str]:
    """Returns a list of failure messages (empty == all good)."""
    url = base_url.rstrip("/") + "/healthz"
    print(f"\n[engine] GET {url}")
    try:
        with urlopen(Request(url), timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        return [f"could not reach {url}: {exc.reason}"]
    except Exception as exc:
        return [f"GET {url} failed: {exc!r}"]

    fails: list[str] = []
    if payload.get("status") != "ok":
        fails.append(f"/healthz status != 'ok': {payload.get('status')!r}")
    if not payload.get("model_loaded"):
        fails.append("/healthz model_loaded is falsey")

    loaded = set(payload.get("banks_loaded", []))
    missing = REQUIRED_PAGE_KEYS - loaded
    if missing:
        fails.append(
            f"banks_loaded missing required page_keys: {sorted(missing)} "
            f"(saw {sorted(loaded)})"
        )
    if not fails:
        _print_ok(
            f"engine healthy, banks_loaded covers {sorted(REQUIRED_PAGE_KEYS)}"
        )
    return fails


def check_banks_on_disk(banks_dir: Path) -> list[str]:
    """Validate byte-length + dtype/shape for every manifest entry."""
    print(f"\n[disk] validating {banks_dir}")
    fails: list[str] = []

    manifest_path = banks_dir / bank_io.MANIFEST_NAME
    if not manifest_path.exists():
        return [f"no manifest at {manifest_path}"]

    manifest = bank_io.read_manifest(str(banks_dir))
    entries = manifest.get("banks", [])
    if not entries:
        return [f"manifest has zero banks: {manifest_path}"]

    page_keys = {e["page_key"] for e in entries}
    missing = REQUIRED_PAGE_KEYS - page_keys
    if missing:
        fails.append(f"manifest missing required page_keys: {sorted(missing)}")

    for entry in entries:
        page_key = entry["page_key"]
        num_slots = int(entry["num_slots"])
        expected_bytes = num_slots * EXPECTED_BYTES_PER_SLOT

        layers_seen: set[int] = set()
        for layer_str, files in entry["files"].items():
            layer = int(layer_str)
            layers_seen.add(layer)
            for kind in ("k", "v"):
                p = banks_dir / files[kind]
                if not p.exists():
                    fails.append(f"{page_key} L{layer} {kind}: missing {p}")
                    continue
                size = os.path.getsize(p)
                if size != expected_bytes:
                    fails.append(
                        f"{page_key} L{layer} {kind}: size {size} != "
                        f"expected {expected_bytes} (num_slots={num_slots})"
                    )

        if layers_seen != set(SELECTED_LAYERS):
            fails.append(
                f"{page_key}: layers {sorted(layers_seen)} != "
                f"{SELECTED_LAYERS}"
            )

        try:
            loaded = bank_io.load_bank(entry, str(banks_dir))
        except Exception as exc:
            fails.append(f"{page_key}: bank_io.load_bank raised: {exc!r}")
            continue
        slot_counts: set[int] = set()
        for layer, (k, v) in loaded.items():
            slot_counts.add(k.shape[1])
            if k.shape != (NUM_KV_HEADS, num_slots, HEAD_DIM):
                fails.append(
                    f"{page_key} L{layer} K shape {k.shape} != "
                    f"({NUM_KV_HEADS},{num_slots},{HEAD_DIM})"
                )
            if v.shape != (NUM_KV_HEADS, num_slots, HEAD_DIM):
                fails.append(
                    f"{page_key} L{layer} V shape {v.shape} != "
                    f"({NUM_KV_HEADS},{num_slots},{HEAD_DIM})"
                )
            if str(k.dtype) != "float32" or str(v.dtype) != "float32":
                fails.append(
                    f"{page_key} L{layer}: dtype "
                    f"k={k.dtype} v={v.dtype} (expected float32)"
                )
        if len(slot_counts) > 1:
            fails.append(
                f"{page_key}: num_slots differs across layers: {slot_counts}"
            )

        if not any(
            f.startswith(page_key) for f in fails
        ):
            _print_ok(
                f"{page_key:<11} S={num_slots:<4} layers={sorted(layers_seen)}"
            )

    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--banks-dir",
        default=str(REPO_ROOT / "banks"),
        help="Directory containing manifest.json + .bin files.",
    )
    parser.add_argument(
        "--inference-url",
        default=os.environ.get("INFERENCE_URL", DEFAULT_INFERENCE_URL),
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Skip the /healthz check (useful when validating off-EC2).",
    )
    args = parser.parse_args()

    fails: list[str] = []

    fails.extend(check_banks_on_disk(Path(args.banks_dir)))

    if args.skip_engine:
        print("\n[engine] skipped (--skip-engine)")
    else:
        fails.extend(check_engine_healthz(args.inference_url))

    print()
    if fails:
        for f in fails:
            _print_err(f)
        print(f"\nFAILED: {len(fails)} issue(s)", file=sys.stderr)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
