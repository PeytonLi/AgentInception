"""CONTRACTS.md §3 — structural hash ignores text/scripts/styles, keeps tags+classes."""

from agentinception_shared.dom_hash import dom_structural_hash


def test_text_changes_do_not_affect_hash():
    a = "<div class='a'><p>Hello world</p></div>"
    b = "<div class='a'><p>Completely different text here</p></div>"
    assert dom_structural_hash(a) == dom_structural_hash(b)


def test_scripts_and_styles_ignored():
    a = "<div class='x'><span>hi</span></div>"
    b = (
        "<div class='x'><script>var z=1;</script>"
        "<style>.x{color:red}</style><span>hi</span></div>"
    )
    assert dom_structural_hash(a) == dom_structural_hash(b)


def test_comments_ignored():
    a = "<ul><li>one</li></ul>"
    b = "<ul><!-- a comment --><li>one</li></ul>"
    assert dom_structural_hash(a) == dom_structural_hash(b)


def test_class_order_normalized():
    a = "<div class='alpha beta'></div>"
    b = "<div class='beta alpha'></div>"
    assert dom_structural_hash(a) == dom_structural_hash(b)


def test_structural_difference_changes_hash():
    a = "<div class='a'></div>"
    b = "<section class='a'></section>"
    assert dom_structural_hash(a) != dom_structural_hash(b)


def test_class_difference_changes_hash():
    a = "<div class='a'></div>"
    b = "<div class='b'></div>"
    assert dom_structural_hash(a) != dom_structural_hash(b)


def test_hash_is_hex_sha256():
    h = dom_structural_hash("<html><body></body></html>")
    assert len(h) == 64
    int(h, 16)  # parses as hex
