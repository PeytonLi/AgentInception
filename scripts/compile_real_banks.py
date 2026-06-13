#!/usr/bin/env python
"""Compile REAL KV banks for all 3 demo page types on a GPU box.

This is the canonical path for producing banks that actually steer the model.
Each bank goes through the full B1 pipeline:

    DOM → Haiku summary → Llama-3.1-8B forward pass → pre-RoPE K/V banks

The model is loaded once and reused across all three page types to avoid
repeated cold starts (~90s per load on A10G).

Prerequisites:
    - GPU box with CUDA (g5.2xlarge / A10G recommended)
    - HF_TOKEN env var (gated Llama-3.1-8B-Instruct access)
    - ANTHROPIC_API_KEY env var (Haiku DOM summaries)
    - `pip install -e apps/bank-compiler` + `pip install -e packages/shared-py`
    - Playwright installed if using --live (for HN capture)

Usage:
    # Compile all 3 page types using snapshots in demo-assets/snapshots/
    python scripts/compile_real_banks.py

    # Capture fresh HN pages first, then compile
    python scripts/compile_real_banks.py --live

    # Compile only one page type
    python scripts/compile_real_banks.py --page-key hn:front

    # Custom output directory
    python scripts/compile_real_banks.py --out /tmp/banks

After compiling, upload to ClickHouse:
    python scripts/upload_banks.py banks/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "bank-compiler" / "src"))

ALL_PAGE_TYPES = {
    "hn:front": {
        "snapshot": "demo-assets/snapshots/hn_front.html",
        "url": "https://news.ycombinator.com/",
        "domain": "news.ycombinator.com",
    },
    "hn:item": {
        "snapshot": "demo-assets/snapshots/hn_item.html",
        "url": "https://news.ycombinator.com/item?id=1",  # placeholder; live captures a real one
        "domain": "news.ycombinator.com",
    },
    "popup:demo": {
        "snapshot": "demo-assets/popup-page/index.html",
        "url": None,  # always from file
        "domain": "localhost",
    },
}


def _check_prerequisites() -> list[str]:
    """Return a list of missing prerequisites (empty = ready to go)."""
    issues: list[str] = []
    if not os.environ.get("HF_TOKEN"):
        issues.append("HF_TOKEN env var not set (required for gated Llama-3.1-8B)")
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get(
        "DEEPSEEK_API_KEY"
    ):
        issues.append(
            "Neither ANTHROPIC_API_KEY nor DEEPSEEK_API_KEY is set (required for DOM summaries)"
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        issues.append("PyTorch not installed")
    try:
        import transformers  # noqa: F401
    except ImportError:
        issues.append("transformers not installed")
    try:
        from bank_compiler.compiler import run_compile  # noqa: F401
    except ImportError:
        issues.append(
            "bank_compiler not importable (pip install -e apps/bank-compiler)"
        )
    return issues


def _capture_live_hn(snapshot_dir: Path) -> None:
    """Capture fresh HN front page and a representative item page."""
    capture_script = REPO_ROOT / "scripts" / "capture_dom.py"
    if capture_script.exists():
        import subprocess

        print("[capture] Running capture_dom.py for fresh HN snapshots...")
        subprocess.run(
            [sys.executable, str(capture_script), "--out", str(snapshot_dir)],
            check=True,
        )
    else:
        print(
            "[capture] WARNING: scripts/capture_dom.py not found. "
            "Using existing snapshots if available.",
            file=sys.stderr,
        )


def compile_all(
    page_keys: list[str],
    out_dir: Path,
    live: bool = False,
) -> int:
    """Compile banks for the given page types. Returns 0 on success."""
    from bank_compiler.cli import validate_dir
    from bank_compiler.compiler import CompileOptions, run_compile
    from bank_compiler.encoder import load_model_and_tokenizer

    snapshot_dir = REPO_ROOT / "demo-assets" / "snapshots"

    # If --live, capture fresh HN pages first.
    if live:
        _capture_live_hn(snapshot_dir)

    # Load the model ONCE (the expensive part: ~90s on A10G).
    print("\n[model] Loading Llama-3.1-8B-Instruct (this takes ~90s on first run)...")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer()
    print(f"[model] Ready in {time.perf_counter() - t0:.1f}s\n")

    results: list[dict] = []
    for pk in page_keys:
        spec = ALL_PAGE_TYPES[pk]
        print(f"[compile] {pk} ...")
        t1 = time.perf_counter()

        # Determine source: file or URL.
        snapshot_path = REPO_ROOT / spec["snapshot"]
        if pk == "popup:demo":
            # Always compile from the local fixture file.
            opts = CompileOptions(
                page_key=pk,
                out_dir=str(out_dir),
                html=str(snapshot_path),
                url=f"http://localhost:8080/popup-page/index.html",
                model=model,
                tokenizer=tokenizer,
            )
        elif snapshot_path.exists() and not live:
            # Use existing snapshot.
            opts = CompileOptions(
                page_key=pk,
                out_dir=str(out_dir),
                html=str(snapshot_path),
                url=spec["url"],
                model=model,
                tokenizer=tokenizer,
            )
        elif live:
            # Live capture (Playwright).
            opts = CompileOptions(
                page_key=pk,
                out_dir=str(out_dir),
                url=spec["url"],
                model=model,
                tokenizer=tokenizer,
            )
        else:
            print(f"  SKIP {pk}: no snapshot at {snapshot_path} and --live not set")
            continue

        entry = run_compile(opts)
        elapsed = time.perf_counter() - t1
        print(
            f"  done {pk:<11}  slots={entry['num_slots']:<4}  "
            f"synthetic={entry.get('synthetic', 'n/a')}  ({elapsed:.1f}s)"
        )
        results.append(entry)

    # Validate the output.
    print(f"\n[validate] Checking {out_dir} ...")
    rc = validate_dir(str(out_dir))
    if rc != 0:
        print("\n[validate] FAILED — see errors above.", file=sys.stderr)
        return rc

    # Summary table.
    print("\n" + "=" * 60)
    print("COMPILATION SUMMARY")
    print("=" * 60)
    for e in results:
        print(
            f"  {e['page_key']:<11}  slots={e['num_slots']:<4}  "
            f"hash={e.get('dom_structural_hash', '?')[:12]}...  "
            f"synthetic={e.get('synthetic', '?')}"
        )
    print(f"\nManifest: {out_dir / 'manifest.json'}")
    print("Next step: python scripts/upload_banks.py " + str(out_dir))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="banks",
        help="Output directory for .bin files + manifest (default: banks/)",
    )
    parser.add_argument(
        "--page-key",
        choices=list(ALL_PAGE_TYPES.keys()),
        action="append",
        dest="page_keys",
        help="Compile only this page type (can repeat; default: all 3)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture fresh HN pages via Playwright before compiling "
        "(requires playwright install chromium).",
    )
    args = parser.parse_args()

    # Check prerequisites.
    issues = _check_prerequisites()
    if issues:
        print("Cannot compile real banks — prerequisites missing:", file=sys.stderr)
        for issue in issues:
            print(f"  ✗ {issue}", file=sys.stderr)
        print(
            "\nSee docs/handoff/phase-2/rahul/R1-real-banks.md for setup.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_keys = args.page_keys or list(ALL_PAGE_TYPES.keys())
    return compile_all(page_keys, out_dir, live=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
