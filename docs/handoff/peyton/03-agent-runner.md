# Agent A3 — Agent Runner (Playwright Loop)

**Owner track:** Peyton
**Builds:** `apps/agent-runner/`
**Reads first:** `docs/handoff/CONTRACTS.md` (§3, §6–§9)
**Depends on:** `shared-py` (A2). Develops fully against `tests/mocks/mock_inference.py` — does NOT wait for the real engine.

## Mission

The Python driver loop: open a browser, walk the demo task, call `/api/v1/step` each iteration, execute the returned action, stream viewport frames, and produce the token metrics that power the demo's comparison chart. Two modes, same loop: `--mode=baseline` and `--mode=mi`.

## Tasks (TDD — write the test before each piece)

1. **Scaffold** `apps/agent-runner/` — `playwright`, `httpx`, `tiktoken`-equivalent? No: use `transformers AutoTokenizer` for Llama tokenizer counts (slow-load once, cache). CLI via `argparse`: `python -m agent_runner --mode=mi --task="..." --start-url=https://news.ycombinator.com --session-id=...`.
2. **Build `tests/mocks/mock_inference.py`** (you own this mock; A4 and C1 reuse it): FastAPI stub on :8000 implementing CONTRACTS §6/§7 with a scripted action sequence for the HN task and a canned WS event loop.
3. **The loop:**
   - Launch Chromium (headed for demo, headless for tests), viewport 1280×720.
   - Per step: read `page.url()` → `page_key()` → extract DOM text (`document.body.innerText`-based, scripts/styles stripped, truncated to ~4,000 Llama tokens) → POST `/api/v1/step` (baseline: include `dom_text`; mi: `dom_text=null` + `dom_token_count` of what baseline WOULD have sent) → execute action → log step to ClickHouse `agent_steps`.
   - Action execution per CONTRACTS §8: `goto`, `click` (selector with 5s timeout + one retry), `dismiss_modal`, `extract`/`done` (print result, end loop). Max 15 steps, hard stop.
4. **Viewport streaming:** background task, every 300ms `page.screenshot(type="jpeg", quality=50)` → base64 → `POST {INFERENCE_URL}/internal/frame`. Fire-and-forget; never block the loop on it.
5. **Metrics:** maintain `cum_visible` and `cum_baseline` counters; include in each step request so the engine can emit `token_metrics`. Baseline mode: both numbers identical by construction (that IS the baseline).
6. **Popup handling is NOT special-cased in the runner.** The model (steered by the popup bank) must return `dismiss_modal`. The runner just executes it. Do not add an `if modal: dismiss` shortcut — that would fake the demo's core claim.

## Unit tests (write first)

- `test_dom_extraction_truncation`: 50k-token fixture page → extracted text ≤ 4,000 tokens, scripts/styles absent.
- `test_action_execution_each_type`: local fixture HTML pages; `click`/`goto`/`dismiss_modal` mutate page state as expected.
- `test_loop_against_mock`: full loop vs `mock_inference` on local fixture pages (snapshot of HN front + item saved into `tests/fixtures/pages/`) completes in N scripted steps and writes N rows to a fake step log.
- `test_malformed_action_retry`: mock returns garbage once → one re-prompt; twice → abort with clear error.
- `test_token_counters_monotonic`: cum counters strictly increase; mi-mode `cum_visible` grows by < 500/step.

## Definition of done

- Both modes run end-to-end against the mock on fixture pages, headless, in CI-able form (`pytest`).
- Against the real engine (post H+4 sync): one full mi-mode run on live HN completes the task; frames visible in the console.

## Suggested skills

`superpowers:test-driven-development`, `everything-claude-code:e2e-testing`, `superpowers:verification-before-completion`
