# Agent P4 - Web Console Against the Real Run (Peyton)

**Owner:** Peyton  **Branch:** `phase2/p4-console-live`  **Worktree:** `.claude/worktrees/phase2-p4`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s7),
`apps/web-console/lib/{events.ts,eventReducer.ts,useEventFeed.ts}`,
`apps/web-console/components/{ViewportPanel,TokenComparator,LayerInjectionGraph,LogsMathPanel}.tsx`,
`tests/mocks/mock_ws_feed.py`, `apps/web-console/e2e/dashboard.spec.ts`.
**Depends on:** P1 (engine WS exists). Can start immediately against the mock feed; swaps
to the real WS once P1 is up. **Largely parallel** to P2/P3. **Unblocks:** P5 (demo).

## Mission

The dashboard renders correctly against `mock_ws_feed.py` but has never seen a real run.
Point it at the live engine WS, make all 4 panels behave under real-event timing and
volume, and bring it to demo polish - this is the surface the judges actually watch.

## Tasks

1. **Real WS wiring.** Set `NEXT_PUBLIC_INFERENCE_WS=ws://<p1-box>:8000/ws/events` and
   run a live `mi` session (driven by P3). Confirm `useEventFeed.ts` consumes the real
   `layer_injection`, `token_metrics`, `action`, `viewport_frame`, `log` events with no
   schema drift from `CONTRACTS.md` s7. Fix any reducer mismatch in `eventReducer.ts`.
2. **Viewport panel under real frames.** Real frames arrive ~300 ms, base64 JPEG. Verify
   no memory leak / lag over a multi-minute run; throttle/drop stale frames if needed.
   Wire the "crimson flash on popup bank" cue to the real `layer_injection` for
   `popup:demo`.
3. **Token comparator under real data.** Two diverging series (baseline climbs, MI flat)
   driven by real `cum_visible`/`cum_baseline`; animate the `kv_savings_ratio`. Make it
   legible at projector resolution.
4. **Layer injection graph.** 32 rows; 8/12/16/20 light up on real `layer_injection`
   events and go dark on `active:false` (the unknown-page fallback moment). Confirm
   timing reads well live.
5. **Logs & math panel.** Scrolling real logs + the live KV ratio + the K*/V* equation.
   Ensure the equation shows the real `num_slots` and `dom_token_count`.
6. **Resilience.** Verify reconnect (`reconnect.test.ts`) works when the engine restarts
   mid-demo; show a clear "reconnecting" state rather than a frozen panel.
7. **Polish + capture.** Apply `frontend-design` judgment (spacing, contrast, motion
   restraint). Take a high-res screenshot of a populated live run for the README/slides
   and save under `docs/handoff/phase-2/notes/p4-console-shot.png`.

## Definition of done

- Console renders a real live run end-to-end: 4 panels populate, layers light up, chart
  shows two diverging series, ratio animates.
- `dashboard.spec.ts` extended with a Playwright test against a recorded real-event WS
  fixture (so it is reproducible in CI without GPU).
- Reconnect verified against an engine restart.
- Screenshot committed.
- Write scope limited to `apps/web-console/` + `docs/handoff/phase-2/notes/`.

## Commit / push / PR

Commit per panel/fix. Push `phase2/p4-console-live`; PR "Phase 2 / P4 - console live".
Attach the screenshot in the PR body.

## Suggested skills

`frontend-design`, `tdd` (reducer + e2e), `git-commit`.
