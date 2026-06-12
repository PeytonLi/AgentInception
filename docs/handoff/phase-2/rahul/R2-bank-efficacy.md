# Agent R2 - Bank Steering-Efficacy Harness & Tuning (Rahul)

**Owner:** Rahul  **Branch:** `phase2/r2-bank-efficacy`  **Worktree:** `.claude/worktrees/phase2-r2`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s4),
`apps/bank-compiler/src/bank_compiler/{summarizer.py,encoder.py}`,
`apps/inference-engine/scripts/prove_injection.py`, paper s3 + Appendix C/G.2.
**Depends on:** P1 (model) and R1 (a real bank to measure). Runs **in parallel** with R1
and feeds quality decisions back into it. **Unblocks:** confidence in P3/P5 demo quality.

## Mission

Right now "is this bank good?" has no answer beyond "the shape is right". Build a small,
reusable **steering-efficacy harness** that quantifies whether a bank actually makes the
model behave as if it saw the page - then use it to tune the Haiku summaries so the demo
banks are convincing, not just present.

## Why this matters

A bank with the correct shape can still be a weak steer if the summary is vague or the
kept-position slice is wrong. P3's demo lives or dies on the model picking the right HN
links from latent memory alone. R2 turns bank quality from a vibe into a number R1 can
optimize against.

## Tasks

1. **Efficacy metric.** In `apps/bank-compiler/` (new `efficacy/` module or
   `scripts/measure_efficacy.py`), define a quantitative steer score for a given bank +
   probe question. Suggested: for a set of held-out probe prompts about the page (e.g.
   "what is the selector to open comments?", "where is the score?"), compare the model's
   answer/logits with the real bank injected vs. with the full DOM in-context vs. with
   no context. A good bank's behavior tracks the full-DOM behavior and diverges from
   no-context. Report KL-to-DOM and KL-to-empty per probe.
2. **Probe sets.** Write 5-8 probe prompts per page type capturing exactly the facts the
   demo task needs (HN front: comments-link selector, More link, score span; HN item:
   score location, top commenters; popup: dismiss selector). Keep them in
   `apps/bank-compiler/efficacy/probes/`.
3. **Harness CLI.** `measure_efficacy.py --page-key hn:front --bank banks/...` prints a
   table: per-probe KL-to-DOM (lower = better), KL-to-empty (higher = better), and a
   single roll-up score. Mark it `@pytest.mark.gpu` where wrapped in a test.
4. **Tune summaries.** Use the harness to iterate `summarizer.py` prompt wording and the
   200-400 word target so each bank's roll-up score clears a documented threshold. Hand
   improved summaries to R1 to recompile (coordinate; R1 owns the actual `.bin` output).
5. **Quality report.** `docs/handoff/phase-2/notes/r2-efficacy.md`: the metric
   definition, per-page scores for the final banks, and a short "what makes a good
   summary" guide for future page types.

## Definition of done

- `measure_efficacy.py` runs on the box and scores a real bank against its probes.
- Each of the 3 demo banks clears the documented efficacy threshold (or the gap is
  explained in the report with a mitigation).
- Probe sets + report committed.
- Write scope limited to `apps/bank-compiler/efficacy/`, `scripts/measure_efficacy.py`,
  `docs/handoff/phase-2/notes/`. Do NOT write `.bin` files - that is R1's job; you hand
  R1 tuned summaries.

## Commit / push / PR

Commit the harness, then the probes, then the report. Push `phase2/r2-bank-efficacy`;
PR "Phase 2 / R2 - bank efficacy". PR body: the final per-page scores table.

## Suggested skills

`tdd`, `diagnose`, Anthropic API docs (summary tuning), `git-commit`.
