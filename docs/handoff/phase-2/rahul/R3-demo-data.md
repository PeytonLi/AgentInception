# Agent R3 - Demo Data, Fixtures & Token-Honesty Dataset (Rahul)

**Owner:** Rahul  **Branch:** `phase2/r3-demo-data`  **Worktree:** `.claude/worktrees/phase2-r3`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s3, s8),
`demo-assets/popup-page/index.html`, `demo-assets/popup-page/tests/test_popup_modal.py`,
`apps/agent-runner/tests/fixtures/pages/{hn_front.html,hn_item.html,popup.html}`,
`docs/handoff/rahul/02-demo-assets-and-data.md`.
**Depends on:** nothing to start (capture work is independent). Coordinates with R1 (feeds
fresh DOM snapshots) and P3 (consumes fixtures). **Unblocks:** R1 quality, P3 reliability.

## Mission

Make the demo's input data real, fresh, and reproducible: capture current HN DOM
snapshots for compilation, harden the local popup fixture page so the chaos test is
deterministic, and build a small "token-honesty" dataset that pins the baseline-vs-MI
token numbers the demo claims.

## Why this matters

Phase-1 banks were compiled from placeholder summaries; the agent-runner fixtures are
stale checked-in HTML. Live HN drifts (story lists, comment counts). R3 gives R1 honest
DOM to compile from and gives P3 stable fixtures so a flaky live site does not sink a
rehearsal.

## Tasks

1. **Fresh DOM capture.** Add `scripts/capture_dom.py` (Playwright) that snapshots the
   live HN front page and a representative HN item page to
   `demo-assets/snapshots/{hn_front,hn_item}.html` plus their extracted text and
   `dom_structural_hash`. Hand these to R1 as the compile source so summaries and hashes
   are real. Re-runnable so the snapshot can be refreshed right before the demo.
2. **Popup fixture hardening.** Polish `demo-assets/popup-page/index.html`: a cookie-
   consent modal with a stable `data-testid` and a dismiss selector matching the
   `dismiss_modal` action in `CONTRACTS.md` s8. Ensure it serves locally (file:// or a
   tiny static server) and that `page_key()` maps it to `popup:demo`. Extend
   `test_popup_modal.py` so the modal's presence/absence is deterministically assertable
   (this is what P3's chaos test keys off).
3. **Token-honesty dataset.** Build `demo-assets/token_honesty/` with, per page type:
   the captured DOM, its Llama-tokenizer token count (the `dom_token_count` baseline
   WOULD send), and the expected `kv_savings_ratio` given the bank's `num_slots`. Add
   `scripts/tests/test_token_honesty.py` asserting the README formula
   `(NUM_LAYERS * dom_token_count) / (L_injected * num_slots)` lands in the honest
   20-80x band - so we never overclaim on stage.
4. **Agent-runner fixtures refresh.** Update `apps/agent-runner/tests/fixtures/pages/*`
   from the fresh snapshots so the runner's offline tests reflect current HN structure.
   Coordinate with P3 (it owns the runner tests) - hand over the HTML, let P3 wire it,
   or do a tiny scoped PR.
5. **Reproducibility note.** `docs/handoff/phase-2/notes/r3-data.md`: how to re-capture,
   token counts per page, and the chosen "pinned" demo story strategy (so the live demo
   targets a known story to reduce variance).

## Definition of done

- `scripts/capture_dom.py` produces fresh `hn_front`/`hn_item` snapshots on demand.
- Popup fixture deterministic; `test_popup_modal.py` asserts modal presence/absence.
- `token_honesty/` dataset + passing `test_token_honesty.py` proving the 20-80x claim.
- Snapshots handed to R1; refreshed runner fixtures handed to P3.
- Write scope: `demo-assets/`, `scripts/capture_dom.py`, `scripts/tests/test_token_honesty.py`,
  `docs/handoff/phase-2/notes/`. (Runner-test edits go through P3.)

## Commit / push / PR

Commit per artifact (capture script, popup, honesty dataset). Push `phase2/r3-demo-data`;
PR "Phase 2 / R3 - demo data". PR body: the token counts table and the pinned-story plan.

## Suggested skills

`tdd`, `frontend-design` (popup page), Playwright/e2e, `git-commit`.
