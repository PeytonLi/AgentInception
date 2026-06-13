"""Orchestrator: DOM extract -> cloud summary -> Llama encode -> bank IO.

This module exposes `run_compile()` for both the CLI and tests. The three
heavy collaborators (DOM loader, summarizer, encoder, model loader) are looked
up by module-level name so tests can monkeypatch them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinception_shared import bank_io

from .dom_extract import domain_of, load_dom
from .encoder import encode_summary, load_model_and_tokenizer
from .provenance import tag_synthetic
from .summarizer import summarize_dom


@dataclass
class CompileOptions:
    page_key: str
    out_dir: str
    url: str | None = None
    html: str | None = None
    selected_layers: list[int] | None = (
        None  # default: agentinception_shared SELECTED_LAYERS
    )
    anthropic_client: Any = None  # for tests (deprecated; use summarizer_client)
    summarizer_client: Any = None  # pre-built client for the summarizer
    summarizer_provider: str | None = (
        None  # 'anthropic' or 'deepseek' (auto-detected if None)
    )
    model: Any = None  # pre-loaded model (skips load_model_and_tokenizer)
    tokenizer: Any = None


def _resolve_url_for_domain(opts: CompileOptions, dom_url: str) -> str:
    # `dom_url` is the canonical URL captured by the extractor (file:// for html=).
    if opts.url:
        return opts.url
    return dom_url


def run_compile(opts: CompileOptions) -> dict[str, Any]:
    """Run the full pipeline and return the manifest entry that was written."""
    if not opts.page_key:
        raise ValueError("page_key is required")
    if opts.html is None and opts.url is None:
        raise ValueError("Pass at least one of url= or html=")
    out_dir = opts.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. DOM extraction.
    extract = load_dom(url=opts.url, html=opts.html)
    canonical_url = _resolve_url_for_domain(opts, extract.url)
    domain = domain_of(canonical_url) or domain_of(extract.url)

    # 2. Summarize DOM via cloud LLM (Haiku or DeepSeek).
    summary = summarize_dom(
        dom_text=extract.text,
        url=canonical_url,
        page_key=opts.page_key,
        client=opts.summarizer_client or opts.anthropic_client,
        provider=opts.summarizer_provider,
    )

    # Persist the summary (judges may want to read it).
    summary_filename = opts.page_key.replace(":", "_") + ".summary.txt"
    summary_path = Path(out_dir) / summary_filename
    summary_path.write_text(summary, encoding="utf-8")

    # 3. Llama forward pass -> banks.
    model = opts.model
    tokenizer = opts.tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()
    banks, _kept = encode_summary(
        model=model,
        tokenizer=tokenizer,
        summary_text=summary,
        selected_layers=opts.selected_layers,
    )

    # 4. Persist .bin files + upsert manifest entry via shared-py.
    entry = bank_io.save_bank(
        out_dir=out_dir,
        page_key=opts.page_key,
        banks=banks,
        meta={
            "domain": domain,
            "dom_structural_hash": extract.dom_structural_hash,
            "summary_text_path": str(summary_path.as_posix()),
        },
    )

    # 5. Tag this bank as REAL (not synthetic noise) in the on-disk manifest.
    #    `save_bank` deliberately doesn't know about provenance — the marker
    #    lives in the manifest as an additive key so existing readers ignore
    #    it. The synthetic fallback in `scripts/build_demo_banks.py` sets the
    #    opposite (`"synthetic": true`).
    tag_synthetic(out_dir, opts.page_key, synthetic=False)
    entry["synthetic"] = False
    return entry
