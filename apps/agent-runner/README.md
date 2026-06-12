# agent-runner (A3)

The Playwright driver loop for GhostBrowser OS. It walks the demo task, calls
the inference engine's `POST /api/v1/step` each iteration, executes the returned
Action, streams viewport frames, and produces the token metrics that power the
demo's comparison chart. One loop, two modes: `baseline` and `mi`.

See `docs/handoff/CONTRACTS.md` (§3, §6–§9) for the interface contracts and
`docs/handoff/peyton/03-agent-runner.md` for the brief.

## Install

```bash
pip install -e packages/shared-py          # the shared contract code
pip install -e apps/agent-runner           # this package
python -m playwright install chromium      # browser for live runs
# optional: exact Llama token counts (gated model -> needs HF_TOKEN)
pip install -e "apps/agent-runner[tokenizer]"
```

Copy `.env.example` to `.env` and set `INFERENCE_URL` / `CLICKHOUSE_URL`.

## Run

```bash
python -m agent_runner --mode=mi \
  --task="Find the top story about AI on the Hacker News front page (scan up to
          2 pages), open its comment page, extract the story score and top 3
          commenter usernames." \
  --start-url=https://news.ycombinator.com \
  --session-id=demo-001 \
  --inference-url=http://localhost:8000

# baseline comparison run
python -m agent_runner --mode=baseline --session-id=demo-001-baseline
```

Flags: `--headed` (visible browser), `--no-frames`, `--no-clickhouse`,
`--max-steps N`, `--verbose`.

## Architecture

| Module | Responsibility |
|---|---|
| `loop.py` | The step loop; orchestrates everything. |
| `browser.py` | `PageDriver` protocol + `PlaywrightPageDriver` (lazy import). |
| `dom.py` | `document.body.innerText`-equivalent extraction + token truncation. |
| `tokenizer.py` | Llama tokenizer with a backend-free heuristic fallback. |
| `actions.py` | Action JSON parsing/validation (CONTRACTS §8). |
| `inference_client.py` | `httpx` client for `/api/v1/step` + `/internal/frame`. |
| `frames.py` | Background 300 ms viewport streamer (fire-and-forget). |
| `metrics.py` | Cumulative `visible`/`baseline` token counters. |
| `steplog.py` | `agent_steps` logging via `ghost_shared.storage` (degrades to in-memory). |
| `cli.py` | `argparse` entrypoint wiring it all together. |

### Design notes

- **Dependency injection everywhere.** The loop takes a `PageDriver`, an
  `InferenceClient`, and a `TokenCounter`, so the whole thing is testable
  headless with no browser, no GPU, no HF token, and no ClickHouse.
- **Graceful degradation.** No `transformers` → heuristic token counter. No
  ClickHouse → in-memory step log. No frame can stall the loop.
- **Popup is not special-cased.** The model (steered by the popup bank) returns
  `dismiss_modal`; the runner just executes it. Adding an `if modal: dismiss`
  shortcut would fake the demo's core claim, so it is deliberately absent.

### Contract note (flag to the team)

Per A3 brief task 5, each `/api/v1/step` request carries `cum_visible` and
`cum_baseline` so the engine can emit `token_metrics` with running totals.
These two fields **extend** CONTRACTS §6's request body (additive, optional).
A1 should read them when emitting `token_metrics`; the mock already does.

## Tests

```bash
cd apps/agent-runner && pytest          # 16 tests, no installs required
```

The mock engine (`tests/mocks/mock_inference.py`, owned by A3 and reused by A4
and C1) implements CONTRACTS §6/§7 and runs in-process via ASGITransport. The
real `PlaywrightPageDriver` is exercised against fixture HTML when Chromium is
available; that test skips cleanly otherwise.
