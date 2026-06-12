"""Placeholder so the test task succeeds before the owning agent fills it in."""

import agentinception_shared


def test_shared_importable():
    assert hasattr(agentinception_shared, "compute_page_key")
