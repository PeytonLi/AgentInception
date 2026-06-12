# R3 - Demo Data & Fixtures Notes

## DOM Capture

### How to re-capture

```bash
# Requires: pip install playwright && playwright install chromium
python scripts/capture_dom.py

# With a specific HN item URL:
python scripts/capture_dom.py --item-url https://news.ycombinator.com/item?id=44210001

# Custom output dir:
python scripts/capture_dom.py --out demo-assets/snapshots
```

Snapshots are saved to `demo-assets/snapshots/` with `.meta.json` sidecars
containing `dom_structural_hash`, text length, and capture timestamp.

### Pinned demo story strategy

For live demos, target a **known story** to reduce variance:
1. Before the demo, run `capture_dom.py` to get a fresh front page
2. Pick the top-ranked story with 50+ comments
3. Note its `item?id=` for the demo script
4. The agent-runner test fixtures use synthetic stories that are stable

## Token Counts

| Page Type | Est. DOM Chars | Est. Tokens (÷4) | Bank Slots | Visible Savings Ratio* |
|---|---|---|---|---|
| `hn:front` | ~12,000 | ~3,000 | 312 | 76.9x |
| `hn:item` | ~16,000 | ~4,000 | 420 | 76.2x |  
| `popup:demo` | ~7,000 | ~1,750 | 180 | 77.8x |

*Visible ratio uses `min(dom_token_count, 4000)` truncation:
`(32 × min(tokens, 4000)) / (4 × slots)`

All ratios land in the **20-80x** honest band with baseline truncation applied.

## Popup Fixture

The popup page at `demo-assets/popup-page/index.html` has been hardened with:
- `data-popup-state` attribute: `pending` → `open` → `dismissed`
- `data-testid` attributes on overlay, modal, and both buttons
- `window.__popupModal` API: `isOpen()`, `state()`, `show()`, `dismiss(via)`
- `popup:dismiss` CustomEvent with `detail.via` ∈ `{'accept', 'reject'}`
- Idempotent dismiss (calling twice is a no-op)
- 300ms scripted delay preserved for realistic modal timing

The agent-runner fixture at `apps/agent-runner/tests/fixtures/pages/popup.html`
mirrors this structure.

## Agent-Runner Fixtures

Updated fixtures in `apps/agent-runner/tests/fixtures/pages/`:
- `hn_front.html`: 5 story rows with proper class structure (athing, subtext, score, hnuser, storylink, morelink)
- `hn_item.html`: Full story header + 5 comments with proper comtr structure and navigation links
- `popup.html`: Mirrors hardened popup fixture with data-testid and lifecycle API
