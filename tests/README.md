# tests/

Cross-component test scaffolding. Layout per CONTRACTS.md §2 and §10.

```
tests/
├── integration/   # cross-component tests + demo rehearsal   (owned by C1)
├── mocks/         # mock_inference.py (A3), mock_ws_feed.py (A4)   (§10)
└── fixtures/      # tiny_bank/ shape-correct random bank for A1     (§10)
```

A2 only creates the directory layout. Each consuming agent fills in its own
mocks/fixtures; C1 owns the integration suite.

Per-component unit tests live next to their code (`packages/shared-py/tests`,
`apps/*/tests`).
