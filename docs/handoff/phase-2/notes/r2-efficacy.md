# R2 - Bank Steering-Efficacy Report

## Metric Definition

### Conditions

For each probe prompt, we run the model under three conditions:

| Condition | Bank Injected? | DOM in Prompt? | Purpose |
|---|---|---|---|
| **Bank** | ✓ | ✗ | Measures what the bank teaches the model |
| **Full-DOM** | ✗ | ✓ | Ground truth — model sees the actual page |
| **No-context** | ✗ | ✗ | Baseline ignorance |

### Metrics

- **KL-to-DOM** (lower = better): `KL(bank_logits || dom_logits)` — how closely the bank-injected distribution matches the full-DOM distribution. A value near 0 means the bank is a good proxy for having the DOM in context.

- **KL-to-empty** (higher = better): `KL(bank_logits || empty_logits)` — how much the bank-injected distribution diverges from no-context. A high value means the bank is providing real information.

- **Roll-up score**: `avg(KL-to-empty) - avg(KL-to-DOM)` — single number summarizing bank quality. Higher is better. A good bank has high divergence from empty (it knows things) and low divergence from DOM (it knows the RIGHT things).

### Threshold

Roll-up score must be ≥ **0.1** for a bank to pass. This threshold is conservative and may be adjusted after initial measurements on real banks.

## Per-Page Scores

_To be filled after running on the GPU box with real banks._

| Page Key | Avg KL-to-DOM | Avg KL-to-Empty | Roll-up | Status |
|---|---|---|---|---|
| `hn:front` | — | — | — | PENDING |
| `hn:item` | — | — | — | PENDING |
| `popup:demo` | — | — | — | PENDING |

## Probe Sets

Each page type has 6-7 probes targeting the exact facts the demo agent needs:

### hn:front (7 probes)
- Comments-link selector, pagination (More link), score span selector
- Story ordering, top nav links, story vs comments navigation, story row class

### hn:item (7 probes)
- Score location, top commenters selector, comment row structure
- Commtext location, comment threading, story header, username extraction

### popup:demo (6 probes)
- Dismiss selector, modal presence detection, accept/reject buttons
- Key statistic value, post-dismiss content, overlay data-testid

## What Makes a Good Summary

Based on the harness design, good bank summaries should:

1. **Be specific about selectors** — mention CSS class names, IDs, and data-testid attributes by name. Vague descriptions ("there's a button") don't help the agent find things.

2. **Describe spatial layout** — where elements are relative to each other (header/nav/main/footer). The agent navigates visually.

3. **Include action hints** — "click the 'X comments' link in the subtext (not the title)" beats "there are links on the page."

4. **Cover the demo task** — the summary should mention exactly the facts the locked demo task needs (score, top commenters for HN; dismiss selector for popup).

5. **Stay in the 200-400 word band** — too short misses critical details; too long creates too many kept positions, inflating the bank size without proportional benefit.

6. **End with behavioral guidance** — for popup:demo, the dismiss instruction should be the last sentence so it's the model's freshest memory.

## How to Run

```bash
# On the GPU box:
python scripts/measure_efficacy.py --page-key hn:front --bank banks/
python scripts/measure_efficacy.py --page-key hn:item --bank banks/
python scripts/measure_efficacy.py --page-key popup:demo --bank banks/ \
    --dom-file demo-assets/popup-page/index.html
```

## Tuning Workflow

1. Run the harness → identify weak probes (high KL-to-DOM)
2. Adjust `summarizer.py` prompt wording to emphasize the facts those probes test
3. Recompile the bank (coordinate with R1)
4. Re-run the harness → confirm improvement
5. Repeat until all banks clear the threshold
