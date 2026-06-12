"""End-to-end compile test using stubs for DOM extractor / summarizer / encoder,
plus a roundtrip through shared-py's bank IO."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from bank_compiler import compiler as compiler_mod
from bank_compiler.compiler import CompileOptions, run_compile
from bank_compiler.dom_extract import DomExtract


def _wire_shaped_banks(num_slots: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in (8, 12, 16, 20):
        k = rng.standard_normal((8, num_slots, 128)).astype(np.float32)
        v = rng.standard_normal((8, num_slots, 128)).astype(np.float32)
        out[L] = (k, v)
    return out


def test_compile_writes_bank_files_summary_and_manifest(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "banks"

    fake_extract = DomExtract(
        url="file:///dummy.html",
        html="<html><body><h1>Hi</h1></body></html>",
        text="Hi",
        dom_structural_hash="a" * 64,
    )

    def fake_dom_loader(url=None, html=None, **_):
        return fake_extract

    def fake_summarize(*, dom_text, url, page_key, client=None):
        return "Summary describing the page in 200 words." + (" filler" * 220)

    num_slots = 13
    banks = _wire_shaped_banks(num_slots)

    def fake_encode(*, model, tokenizer, summary_text, selected_layers):
        return banks, list(range(num_slots))

    monkeypatch.setattr(compiler_mod, "load_dom", fake_dom_loader)
    monkeypatch.setattr(compiler_mod, "summarize_dom", fake_summarize)
    monkeypatch.setattr(compiler_mod, "encode_summary", fake_encode)
    monkeypatch.setattr(compiler_mod, "load_model_and_tokenizer", lambda: (None, None))

    opts = CompileOptions(
        url="https://news.ycombinator.com/",
        page_key="hn:front",
        out_dir=str(out_dir),
    )
    entry = run_compile(opts)

    # Manifest entry shape.
    assert entry["page_key"] == "hn:front"
    assert entry["domain"] == "news.ycombinator.com"
    assert entry["num_slots"] == num_slots
    assert set(entry["files"].keys()) == {"8", "12", "16", "20"}

    # All 8 .bin files exist and have the contract byte length.
    expected_bytes = 8 * num_slots * 128 * 4
    for layer_files in entry["files"].values():
        for name in layer_files.values():
            p = out_dir / name
            assert p.exists()
            assert p.stat().st_size == expected_bytes

    # Summary file written and points where the manifest says.
    summary_path = out_dir / Path(entry["summary_text_path"]).name
    assert summary_path.exists()
    assert "Summary describing" in summary_path.read_text()

    # Manifest on disk matches what run_compile returned.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert any(b["page_key"] == "hn:front" for b in manifest["banks"])


def test_roundtrip_via_shared_io(tmp_path: Path):
    """save_bank then load_bank produces byte-identical arrays."""
    from agentinception_shared import bank_io

    num_slots = 7
    banks = _wire_shaped_banks(num_slots)
    summary = tmp_path / "x.summary.txt"
    summary.write_text("summary")

    entry = bank_io.save_bank(
        out_dir=str(tmp_path),
        page_key="popup:demo",
        banks=banks,
        meta={
            "domain": "localhost",
            "dom_structural_hash": "f" * 64,
            "summary_text_path": str(summary),
        },
    )
    loaded = bank_io.load_bank(entry, str(tmp_path))
    for L, (k, v) in banks.items():
        k2, v2 = loaded[L]
        assert np.array_equal(k, k2)
        assert np.array_equal(v, v2)


def test_validate_reports_problems(tmp_path: Path, capsys):
    from bank_compiler.cli import validate_dir

    # No manifest at all -> failure.
    rc = validate_dir(str(tmp_path))
    assert rc != 0
    out = capsys.readouterr().out
    assert "manifest" in out.lower()


def test_validate_passes_on_good_dir(tmp_path: Path, capsys):
    from bank_compiler.cli import validate_dir
    from agentinception_shared import bank_io

    num_slots = 5
    banks = _wire_shaped_banks(num_slots)
    bank_io.save_bank(
        out_dir=str(tmp_path),
        page_key="hn:front",
        banks=banks,
        meta={"domain": "news.ycombinator.com"},
    )
    rc = validate_dir(str(tmp_path))
    assert rc == 0
