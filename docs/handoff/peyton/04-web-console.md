# Agent A4 — Web Console (Next.js Dashboard)

**Owner track:** Peyton
**Builds:** `apps/web-console/`
**Reads first:** `docs/handoff/CONTRACTS.md` (§7, §9)
**Depends on:** nothing. Develops fully against `tests/mocks/mock_ws_feed.py` (built by A3; if it doesn't exist yet, write a 20-line WS replay server yourself from CONTRACTS §7 and hand it to A3 later).

## Mission

The judge-facing 4-panel dashboard. Single page, dark theme, every panel driven live by the `/ws/events` feed. This is what wins or loses the demo visually — favor polish on the layer graph and the savings chart over feature count.

## Layout (locked)

```
+--------------------------------------+----------------------------------+
| LIVE VIEWPORT MIRROR                 | TOKEN COST COMPARATOR            |
| <img> from viewport_frame events     | two cumulative series:           |
| crimson flash overlay on popup bank  |   baseline (climbs) vs MI (flat) |
+--------------------------------------+----------------------------------+
| LAYER INJECTION GRAPH                | LOGS & MATH                      |
| 32 layer rows; 8/12/16/20 light up   | scrolling log events +           |
| on layer_injection active=true       | KV-ratio badge + K*/V* equation  |
+--------------------------------------+----------------------------------+
```

## Tasks (TDD where it pays — test the reducer, not the pixels)

1. **WS client hook** `useEventFeed()`: connects to `NEXT_PUBLIC_INFERENCE_WS`, auto-reconnect w/ backoff, dispatches typed events into a single `useReducer` store. The reducer is pure and unit-testable — this is where your tests live.
2. **Viewport panel:** render latest `viewport_frame` as `<img src="data:image/jpeg;base64,...">`. When a `layer_injection` event arrives with `page_key === "popup:demo"`, flash a crimson border overlay for 1.5s.
3. **Token comparator:** cumulative line/bar chart (`recharts`) fed by `token_metrics`; big animated counter for `kv_savings_ratio`. Two series legend: "Standard prompting" vs "AgentInception MI".
4. **Layer injection graph:** 32 horizontal layer bars; on `layer_injection active=true`, animate the listed layers to a lit state with `num_slots` label; `active=false` → all idle. CSS transitions, no chart lib needed.
5. **Logs & math panel:** virtualized scrolling list of `log` events; static equation block `K* = [K_prompt ∥ K_bank], V* = [V_prompt ∥ V_bank]` (KaTeX or pre-rendered) with a live "injected at L ∈ {8,12,16,20}" annotation; the savings badge.
6. **Demo controls header:** session id display + a mode indicator (baseline/mi) inferred from `token_metrics.mode`. No control buttons needed — runs are started from the agent-runner CLI.

## Unit tests (write first)

- `eventReducer.test.ts`: each CONTRACTS §7 event type updates state correctly; unknown types ignored; cumulative counters track `token_metrics`.
- `reconnect.test.ts`: socket drop → backoff retry → state preserved.
- One Playwright smoke (`pnpm test:e2e`): page renders all 4 panels against the mock feed; layer bars light up within 2s of a `layer_injection` event.

## Definition of done

- `pnpm dev` + mock feed = fully animated dashboard with zero real backend.
- Reducer tests green.
- Swapping `NEXT_PUBLIC_INFERENCE_WS` to the EC2 URL is the only change needed for the real demo.

## Suggested skills

`frontend-design:frontend-design`, `everything-claude-code:nextjs-turbopack`, `superpowers:test-driven-development`
