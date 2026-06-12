# web-console (Agent A4)

Judge-facing 4-panel dashboard for **GhostBrowser OS**. Single dark page, every
panel driven live by the inference engine's `/ws/events` feed (CONTRACTS.md §7).

```
+------------------------------+------------------------------+
| LIVE VIEWPORT MIRROR         | TOKEN COST COMPARATOR        |
|  viewport_frame -> <img>     |  baseline (climbs) vs MI     |
|  crimson flash on popup bank |  animated kv_savings_ratio   |
+------------------------------+------------------------------+
| LAYER INJECTION GRAPH        | LOGS & MATH                  |
|  32 rows; 8/12/16/20 light   |  scrolling logs + K*/V* eqn  |
|  up on layer_injection       |  + KV-ratio badge            |
+------------------------------+------------------------------+
```

## Develop

```bash
pnpm install
cp .env.example .env.local        # default points at the local mock feed
pnpm dev                          # http://localhost:3000
```

Drive it with zero real backend using the mock feed:

```bash
pip install websockets
python ../../tests/mocks/mock_ws_feed.py   # ws://localhost:8000/ws/events
```

For the real demo, the **only** change is `NEXT_PUBLIC_INFERENCE_WS` ->
`ws://<ec2-ip>:8000/ws/events`.

## Test

```bash
pnpm test        # vitest: pure reducer + reconnect/backoff (the logic core)
pnpm test:e2e    # playwright smoke: 4 panels render, layers light up < 2s
```

## Architecture

- `lib/events.ts` — typed CONTRACTS §7 event schema + `parseEvent` guard.
- `lib/eventReducer.ts` — **pure**, unit-tested store. All panels read from it.
- `lib/useEventFeed.ts` — WS client: auto-reconnect w/ exponential backoff,
  injectable socket factory for tests. State survives reconnects.
- `components/*` — one component per panel; presentational, fed by reducer state.
