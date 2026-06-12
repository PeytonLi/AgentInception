# P3 transcripts

Recorded agent-runner transcripts for the live-HN end-to-end task, in both
`mi` and `baseline` modes. Schema is identical across sources so P5 can swap the
real-engine artifacts in without touching the console.

## Files

| File | Engine | DOM | Reproduce |
|---|---|---|---|
| `mi-live-dom.json` / `baseline-live-dom.json` | mock (scripted) | **real live HN** | `python apps/agent-runner/scripts/record_transcript.py --source live-dom` |
| `mi-mock.json` / `baseline-mock.json` | mock (scripted) | bundled fixtures | `python apps/agent-runner/scripts/record_transcript.py --source mock` (offline) |
| `mi-live.json` / `baseline-live.json` | **real GPU engine** | real live HN | `INFERENCE_URL=http://<p1-box>:8000 pytest tests/test_e2e_real_engine.py -m gpu` |

The `*-live.json` pair is produced on the P1 GPU box by the gated e2e tests; it
is the canonical demo artifact. The `*-live-dom.json` pair is the honest stand-in
generated off-box: the **engine** is still scripted, but the **token accounting**
is real because the page bodies are the live HN front + a real comment page, so
`cum_baseline` is the genuine ~14k-token cost a baseline agent would pay.

## Money shot (live-dom, heuristic token counter)

| Metric | `mi` | `baseline` |
|---|---|---|
| steps to terminal action | 2 | 2 |
| `cum_visible` | **306** | 14,301 |
| `cum_baseline` | **14,301** | 14,301 |
| observed ratio (`cum_baseline/cum_visible`) | **46.7x** | 1.0x (honesty control) |
| structural ratio (densest page) | 250.8x | n/a |

`baseline` carries the full DOM every step, so `cum_visible == cum_baseline`
(1.0x) — exactly the honesty control the brief asks for. `mi` keeps the visible
prompt flat (~150 tokens/step) while the bank does the steering.

Token-honesty check (brief task 3): `cum_visible` (306) < 1,500 and
`cum_baseline` (14,301) > 10x that. The real Llama tokenizer shifts the absolute
counts slightly but not the order of magnitude.

## Structural ratio = the README formula, not a fudge

```
kv_savings_ratio (structural) = (NUM_LAYERS * T_guidance) / (L_injected * S_bank)
```

Inputs are tracked from real steps (`agent_runner/metrics.py`):

- `NUM_LAYERS = 32`
- `T_guidance` = DOM token count baseline would have sent (runner-computed)
- `L_injected = 4` (layers [8, 12, 16, 20], from the engine's `injected_layers`)
- `S_bank` = bank `num_slots` (engine response, else `banks/manifest.json`)

For the canonical HN front page (`T_guidance ≈ 14,000`, `S_bank = 312`):

```
(32 * 14000) / (4 * 312) ≈ 359x
```

which matches the root `README.md` headline exactly. The harness prints this
projection on every run.
