"""DOM extractor tests — HTML path only (Playwright path is exercised in B2)."""

from __future__ import annotations

from pathlib import Path

from bank_compiler.dom_extract import extract_from_html, strip_dom


SAMPLE = """
<!doctype html>
<html><head>
  <title>Hello</title>
  <style>.a{color:red}</style>
  <script>alert(1)</script>
</head>
<body>
  <header><h1>Headline</h1></header>
  <main>
    <article class="post"><p>First paragraph.</p><p>Second paragraph.</p></article>
  </main>
  <!-- a tracking comment -->
  <script>tracking()</script>
  <footer>Footer text</footer>
</body></html>
""".strip()


def test_strip_dom_removes_scripts_styles_and_comments():
    stripped_html, text = strip_dom(SAMPLE)
    assert "<script" not in stripped_html.lower()
    assert "<style" not in stripped_html.lower()
    assert "alert(1)" not in stripped_html
    assert "tracking()" not in stripped_html
    assert "a tracking comment" not in stripped_html

    # Visible text survives.
    assert "Headline" in text
    assert "First paragraph." in text
    assert "Footer text" in text


def test_extract_from_html_returns_dict_with_url_and_hash(tmp_path: Path):
    p = tmp_path / "page.html"
    p.write_text(SAMPLE, encoding="utf-8")
    res = extract_from_html(str(p))
    assert isinstance(res.html, str) and "Headline" in res.html
    assert isinstance(res.text, str) and "First paragraph." in res.text
    assert isinstance(res.dom_structural_hash, str) and len(res.dom_structural_hash) == 64
    assert res.url.startswith("file://")
