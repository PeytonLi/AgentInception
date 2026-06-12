#!/usr/bin/env python
# ======================================================================
#  WARNING --- SYNTHETIC / SHAPE-ONLY FALLBACK ONLY ---  NOT A REAL BANK
# ======================================================================
#
#  This script writes RANDOM NOISE that happens to have the right
#  [8, num_slots, 128] float32 shape per CONTRACTS.md §4. The bytes are
#  meaningless. They will NOT steer the model. They WILL pass byte-length
#  and dtype invariants, which is exactly enough to keep CI green and to
#  let downstream pipeline pieces (upload_banks.py, validate_banks_against
#  _engine.py, the byte-length test) run off-GPU.
#
#  Real demo banks must come from the B1 / R1 compiler on a GPU box
#  with HF_TOKEN + ANTHROPIC_API_KEY:
#
#      python scripts/compile_real_banks.py        # all 3 page types
#      # or, one at a time:
#      python -m bank_compiler compile --url <...> --page-key <...> --out banks/
#
#  Use cases for THIS script:
#    - CI / off-GPU dev: produce a manifest + .bin set that exercises
#      the upload, validate, and engine-load paths without a model.
#    - Local smoke tests of bank_io round-trips and byte-length math.
#
#  Do NOT use this script to populate ClickHouse / S3 for a demo. Any
#  bank produced here is tagged `"synthetic": true` in manifest.json so
#  `python -m bank_compiler validate` warns visibly. The real compile
#  path writes `"synthetic": false`.
# ======================================================================
"""Produce the 3 demo banks (hn:front, hn:item, popup:demo) into banks/
as SYNTHETIC, shape-only artifacts.

For the canonical real-bank pipeline see ``scripts/compile_real_banks.py``
and the R1 spec at ``docs/handoff/phase-2/rahul/R1-real-banks.md``.

When B1's compiler is importable AND HF_TOKEN + ANTHROPIC_API_KEY are
present, this script refuses to run (returns exit code 2) so nobody
accidentally clobbers real banks with noise. Pass ``--force-synthetic``
to override and write shape-only banks anyway (CI-style).

Bin files written this way are gitignored. Summary .txt files and
manifest.json are committed.

Usage:
    python scripts/build_demo_banks.py                  # write to banks/
    python scripts/build_demo_banks.py --out tmp/banks  # custom dir
    python scripts/build_demo_banks.py --force-synthetic
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))

from agentinception_shared import bank_io  # noqa: E402
from agentinception_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS  # noqa: E402
from agentinception_shared.dom_hash import dom_structural_hash  # noqa: E402

# (page_key, domain, num_slots, seed, summary, source-html-for-hash)
DEMO_SPECS = [
    (
        "hn:front",
        "news.ycombinator.com",
        312,
        1,
        (
            "Hacker News front page. A vertically stacked list of ~30 numbered "
            "story rows. Each row has a title link, a small site domain in "
            "parentheses, and a subtext line with the score, submitter, age, "
            "and a 'comments' link of the form item?id=<n>. The top nav bar "
            "has 'new', 'past', 'comments', 'ask', 'show', 'jobs', 'submit'. "
            "To open the discussion for a story, click its 'X comments' link "
            "in the subtext (not the title — that goes to the external "
            "article). To advance the list, click the 'More' link at the "
            "bottom. There is no infinite scroll. Story scores live in the "
            "subtext span with class 'score'. Top-of-page is the highest-"
            "ranked story; lower rows are lower rank.\n"
        ),
        "<html><body><table><tr class='athing'><td>1</td>"
        "<td><a class='titlelink'>Story</a></td></tr></table></body></html>",
    ),
    (
        "hn:item",
        "news.ycombinator.com",
        420,
        2,
        (
            "Hacker News item (comment) page. The top of the page is a single "
            "story summary block: title (a.titlelink), domain, then a subtext "
            "row with the score (span.score), submitter, age, and a 'hide' "
            "link. Below that is a flat-rendered but indented comment tree "
            "with ~50+ comment rows. Each comment row (tr.athing.comtr) has a "
            "header line ('user | age | parent | prev | next') and the "
            "comment body in div.commtext. Commenter usernames live in "
            "a.hnuser within each comment's header. To get the top commenters, "
            "scan a.hnuser elements in document order, dedupe, and take the "
            "first three. The story score is in the span.score at the top of "
            "the page. Avoid expanding collapsed comment threads — they "
            "complicate the selectors and aren't needed for score or top "
            "commenters.\n"
        ),
        "<html><body><table><tr class='athing'></tr>"
        "<tr class='athing comtr'></tr></table></body></html>",
    ),
    (
        "popup:demo",
        "localhost",
        180,
        3,
        (
            "Local fixture page styled as a 'TechWire News' article. A single "
            "article container holds an h1 title, a meta line "
            "(date | author | section), four body paragraphs, and one "
            "highlighted statistic box (div.highlight-stat, id "
            "'key-statistic') containing the extractable fact: '94% energy "
            "efficiency'. On page load a cookie-consent modal "
            "(data-testid 'cookie-modal-overlay') overlays the content with a "
            "dark backdrop. It has two buttons: '#accept-cookies' and "
            "'#reject-cookies'. Either click dismisses the overlay and "
            "reveals the article. The modal blocks all interaction with the "
            "article until dismissed. If a cookie-consent or marketing modal "
            "is blocking the page, dismiss it via its accept/close button, "
            "then resume the original task.\n"
        ),
        Path(REPO_ROOT, "demo-assets", "popup-page", "index.html").read_text(
            encoding="utf-8"
        ),
    ),
]


def _make_synthetic_bank(num_slots: int, seed: int):
    rng = np.random.default_rng(seed)
    banks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, layer in enumerate(SELECTED_LAYERS):
        k = rng.standard_normal(
            (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
        ) * 0.02
        v = rng.standard_normal(
            (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
        ) * 0.02
        banks[layer] = (k, v)
    return banks


def _real_compiler_available() -> bool:
    if not os.environ.get("HF_TOKEN") or not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        importlib.import_module("bank_compiler")
    except ImportError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="banks")
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip the real B1 compiler even if it is importable.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_synthetic and _real_compiler_available():
        print(
            "ERROR: real bank-compiler is importable AND API keys are set; "
            "this synthetic fallback script should NOT be used in that case. "
            "Run `scripts/compile_real_banks.py` or "
            "`python -m bank_compiler compile ...` for real banks.\n"
            "Pass --force-synthetic to override (CI/dev use only).",
            file=sys.stderr,
        )
        return 2

    for page_key, domain, num_slots, seed, summary, html in DEMO_SPECS:
        slug = page_key.replace(":", "_")
        summary_path = out_dir / f"{slug}.summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        banks = _make_synthetic_bank(num_slots=num_slots, seed=seed)
        meta = {
            "domain": domain,
            "dom_structural_hash": dom_structural_hash(html),
            "summary_text_path": str(
                Path("banks") / summary_path.name
            ).replace("\\", "/"),
        }
        entry = bank_io.save_bank(str(out_dir), page_key, banks, meta=meta)
        print(
            f"  wrote {page_key:<11}  slots={entry['num_slots']:<4}  "
            f"summary={summary_path.name}  (SYNTHETIC)"
        )

    # Tag every bank we just wrote as synthetic so downstream code can warn.
    manifest = bank_io.read_manifest(str(out_dir))
    for b in manifest.get("banks", []):
        b["synthetic"] = True
    bank_io.write_manifest(str(out_dir), manifest)

    print(f"\nManifest: {out_dir / 'manifest.json'}")
    print("Note: .bin files are SYNTHETIC noise (gitignored) and every "
          "manifest entry is tagged `synthetic: true`. Real banks come from "
          "`scripts/compile_real_banks.py` on a GPU box.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
