# Agent B2 — Demo Assets & Bank Data

**Owner track:** Rahul
**Builds:** `demo-assets/popup-page/`, the 3 production banks in `banks/`, upload + validation scripts
**Reads first:** `docs/handoff/CONTRACTS.md` (§3–§5)
**Depends on:** B1's compiler for the bank-production tasks (the popup page itself has zero dependencies — build it first while B1 is still in progress).

## Mission

Everything the demo consumes as data: the chaos-test fixture page, the three compiled banks, and the script that loads them into ClickHouse on the EC2 box.

## Tasks

### 1. Popup fixture page (no dependencies — do this first, ~45 min)

`demo-assets/popup-page/index.html` + minimal CSS/JS, served by `python -m http.server 8080` (document the command in the file header):

- Looks like a plausible article/content page (a fake "TechWire News" article is fine — any static content).
- On load (300ms delay), a **cookie-consent modal** covers the content: dark overlay, "We value your privacy", an `#accept-cookies` button and a `#reject-cookies` button. Clicking either removes the overlay and reveals the content.
- The underlying page contains one extractable fact (e.g. a highlighted statistic) so the agent has something to do after dismissing.
- No frameworks, no build step. One HTML file, deterministic every load.
- A `data-testid` on the modal root so tests can assert its presence/absence.

### 2. Compile the three banks (after B1's compiler is done)

```
python -m bank_compiler compile --url https://news.ycombinator.com           --page-key hn:front   --out banks/
python -m bank_compiler compile --url https://news.ycombinator.com/item?id=<pick a busy story> --page-key hn:item --out banks/
python -m bank_compiler compile --html demo-assets/popup-page/index.html     --page-key popup:demo --out banks/
python -m bank_compiler validate banks/
```

- Pick an `hn:item` page with 50+ comments so the summary describes the comment-tree structure well.
- Read all three summary `.txt` files yourself. If a summary misdescribes the page (wrong region names, hallucinated elements), regenerate — bad summaries make weak banks and there is no later step that catches this.
- Commit `banks/manifest.json` and the `.summary.txt` files; the `.bin` files are gitignored — transfer them by `scp`.

### 3. Upload to ClickHouse — `scripts/upload_banks.py`

- Reads `banks/manifest.json`, loads each `.bin` via `shared-py.bank_io`, inserts into `agentinception.latent_memory_banks` per CONTRACTS §5. Idempotent (delete-then-insert by `page_key`).
- Run it on the EC2 box after `scp -r banks/ ec2:...`. Verify with `SELECT page_key, layer_id, num_slots, length(k_bank) FROM agentinception.latent_memory_banks` — `length(k_bank)` must equal `8 * num_slots * 128 * 4`.

### 4. Shape validation script — `scripts/validate_banks_against_engine.py`

- Hits the engine's `/healthz` and asserts all 3 page_keys appear in `banks_loaded`.
- For each bank: byte-length arithmetic check above, plus dtype/shape via `bank_io` load.

## Unit tests (write first)

- `test_popup_modal_behavior` (Playwright): modal present on load; clicking `#accept-cookies` removes it; the extractable fact is then visible.
- `test_upload_idempotent`: run upload twice against a local ClickHouse → row count unchanged.
- `test_byte_length_invariant`: for every manifest entry, stored blob length == `8 * num_slots * 128 * 4`.

## Definition of done

- All 3 banks in ClickHouse on the EC2 box; engine `/healthz` lists all 3.
- Popup page served locally, modal test green.
- The three summary files read as accurate descriptions of their pages (human check).

## Suggested skills

`superpowers:test-driven-development`, `everything-claude-code:clickhouse-io`, `superpowers:verification-before-completion`
