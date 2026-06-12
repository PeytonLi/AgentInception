"""GhostBrowser OS - agent-runner (A3).

The Playwright-driven driver loop. Each iteration it:
  - reads the current page URL -> page_key(),
  - extracts (and, in baseline mode, sends) the DOM text,
  - calls the inference engine's POST /api/v1/step,
  - executes the returned Action (CONTRACTS.md s8),
  - streams viewport frames, and
  - accumulates the token metrics that power the demo comparison chart.

Two modes share one loop: ``--mode=baseline`` and ``--mode=mi``.
"""

from __future__ import annotations

__version__ = "0.1.0"
