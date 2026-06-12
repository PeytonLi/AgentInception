"""Placeholder so the test task succeeds before the owning agent fills it in."""

import ghost_shared


def test_shared_importable():
    assert hasattr(ghost_shared, "compute_page_key")
