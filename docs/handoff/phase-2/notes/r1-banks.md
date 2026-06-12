# R1 - Real Bank Compilation Notes

## Status

Pipeline code is complete and validated off-GPU. Banks on disk are currently
**synthetic** (shape-correct noise). Real compilation requires running
`scripts/compile_real_banks.py` on the P1 GPU box.

## Compilation Pipeline

```
DOM snapshot → Haiku summary → Llama-3.1-8B forward pass → pre-RoPE K/V banks
                                                          (4 layers × [8, S, 128] f32)
```

## Per-Bank Details

| Page Key | Source DOM | Haiku Summary Length | `num_slots` | Status |
|---|---|---|---|---|
| `hn:front` | `demo-assets/snapshots/hn_front.html` | TBD (200-400 words) | TBD (synthetic: 312) | SYNTHETIC |
| `hn:item` | `demo-assets/snapshots/hn_item.html` | TBD (200-400 words) | TBD (synthetic: 420) | SYNTHETIC |
| `popup:demo` | `demo-assets/popup-page/index.html` | TBD (200-400 words) | TBD (synthetic: 180) | SYNTHETIC |

## Commands

```bash
# On the P1 GPU box:
cd ~/agentinception

# 1. Compile all 3 page types (model loaded once)
python scripts/compile_real_banks.py

# 2. Or compile with fresh HN capture
python scripts/compile_real_banks.py --live

# 3. Validate
python -m bank_compiler validate banks/

# 4. Upload to ClickHouse
python scripts/upload_banks.py banks/

# 5. Verify engine sees them
curl http://localhost:8000/healthz
```

## Provenance Tagging

- Real banks: `"synthetic": false` in `manifest.json` (set by `compiler.py`)
- Synthetic banks: `"synthetic": true` (set by `build_demo_banks.py`)
- `python -m bank_compiler validate` warns on synthetic banks

## H+4 Shape Sync Checklist

- [ ] First real `hn:front` bank compiled
- [ ] `validate_banks_against_engine.py` exits 0
- [ ] P2 confirms logit shift (KL > 1e-3) with real bank
- [ ] `clear_bank()` restores bit-exact baseline

## Notes

_Fill in after real compilation on the GPU box._
