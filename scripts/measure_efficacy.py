#!/usr/bin/env python
"""Bank steering-efficacy CLI: measure how well a bank steers the model.

Compares the model's next-token logit distribution under three conditions:
  1. Bank injected (no DOM in prompt)
  2. Full DOM in prompt (ground truth)
  3. No context (baseline ignorance)

A good bank's distribution tracks full-DOM and diverges from no-context.

Metrics per probe:
  - KL-to-DOM  (lower = better): bank matches full-DOM behavior
  - KL-to-empty (higher = better): bank adds real information
  - Roll-up = avg(KL-to-empty) - avg(KL-to-DOM)  (higher = better)

Prerequisites:
    - GPU box with the model loaded
    - pip install -e apps/bank-compiler -e packages/shared-py
    - pip install -e apps/inference-engine  (for MI attention)

Usage:
    python scripts/measure_efficacy.py --page-key hn:front --bank banks/
    python scripts/measure_efficacy.py --page-key popup:demo --bank banks/ --dom-file demo-assets/popup-page/index.html
    python scripts/measure_efficacy.py --page-key hn:front --bank banks/ --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "bank-compiler" / "src"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "bank-compiler"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "inference-engine" / "src"))

DEFAULT_THRESHOLD = 0.1  # minimum roll-up score to consider a bank "good"


def _load_dom_text(page_key: str, dom_file: str | None) -> str:
    """Load DOM text for the ground-truth condition."""
    if dom_file:
        return Path(dom_file).read_text(encoding="utf-8")

    # Try standard locations.
    candidates = [
        REPO_ROOT / "demo-assets" / "snapshots" / f"{page_key.replace(':', '_')}.html",
        REPO_ROOT / "demo-assets" / "popup-page" / "index.html" if page_key == "popup:demo" else None,
    ]
    for c in candidates:
        if c and c.exists():
            return c.read_text(encoding="utf-8")

    print(f"WARNING: no DOM file found for {page_key}; KL-to-DOM will be NaN",
          file=sys.stderr)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--page-key", required=True,
        choices=["hn:front", "hn:item", "popup:demo"],
        help="Page type to measure.",
    )
    parser.add_argument(
        "--bank", default="banks/",
        help="Directory containing manifest.json + .bin files.",
    )
    parser.add_argument(
        "--dom-file",
        help="Path to DOM HTML file for ground-truth comparison.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum roll-up score (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of top logits to compare (default: 100).",
    )
    args = parser.parse_args()

    # Check prerequisites.
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        return 1

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
    except ImportError:
        print("ERROR: torch/transformers not installed.", file=sys.stderr)
        return 1

    from ghost_shared import bank_io
    from bank_compiler.encoder import load_model_and_tokenizer
    from efficacy.harness import load_probes, score_bank

    # Load bank.
    manifest = bank_io.read_manifest(args.bank)
    entry = None
    for e in manifest.get("banks", []):
        if e["page_key"] == args.page_key:
            entry = e
            break
    if not entry:
        print(f"ERROR: no bank for {args.page_key} in {args.bank}/manifest.json",
              file=sys.stderr)
        return 1

    bank_data = bank_io.load_bank(entry, args.bank)

    # Convert to torch tensors for MI attention.
    import torch
    bank_layers = {}
    for layer_id, (k_np, v_np) in bank_data.items():
        bank_layers[layer_id] = (
            torch.from_numpy(k_np),
            torch.from_numpy(v_np),
        )

    # Load model.
    print(f"[efficacy] Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # Load probes.
    probes = load_probes(args.page_key)
    print(f"[efficacy] Loaded {len(probes)} probes for {args.page_key}")

    # Load DOM text.
    dom_text = _load_dom_text(args.page_key, args.dom_file)

    # Score.
    print(f"[efficacy] Scoring bank {args.page_key}...\n")
    result = score_bank(
        model=model,
        tokenizer=tokenizer,
        bank_layers=bank_layers,
        probes=probes,
        dom_text=dom_text,
        top_k=args.top_k,
    )

    # Print results table.
    print(f"\n{'='*70}")
    print(f"EFFICACY REPORT: {args.page_key}")
    print(f"{'='*70}")
    print(f"{'Probe':<40} {'KL-to-DOM':>10} {'KL-to-Empty':>12}")
    print(f"{'-'*40} {'-'*10} {'-'*12}")
    for p in result["per_probe"]:
        kl_dom = f"{p['kl_to_dom']:.4f}" if p['kl_to_dom'] == p['kl_to_dom'] else "N/A"
        kl_empty = f"{p['kl_to_empty']:.4f}" if p['kl_to_empty'] == p['kl_to_empty'] else "N/A"
        print(f"{p['description']:<40} {kl_dom:>10} {kl_empty:>12}")
    print(f"{'-'*40} {'-'*10} {'-'*12}")
    print(f"{'AVERAGE':<40} {result['avg_kl_to_dom']:>10.4f} {result['avg_kl_to_empty']:>12.4f}")
    print(f"\nRoll-up score: {result['rollup_score']:.4f}  (threshold: {args.threshold})")

    if result["rollup_score"] != result["rollup_score"]:  # NaN check
        print("\nWARNING: roll-up is NaN — missing DOM or bank data.", file=sys.stderr)
        return 1

    if result["rollup_score"] >= args.threshold:
        print(f"\n✓ PASS: roll-up {result['rollup_score']:.4f} >= {args.threshold}")
        return 0
    else:
        print(f"\n✗ FAIL: roll-up {result['rollup_score']:.4f} < {args.threshold}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
