"""Bank compiler (B1): offline KV-bank compilation pipeline.

Pipeline (paper §3.2, Eq. 6):

    URL or HTML
        |  dom_extract.load_dom
    DOM text + structural hash
        |  summarizer.summarize_dom (Haiku, 200-400 words)
    Page-structure summary
        |  encoder.encode_summary  (Llama forward pass, no RoPE)
    {layer -> (K, V)} canonical pre-RoPE banks
        |  agentinception_shared.bank_io.save_bank
    banks/<page_key>__L*.bin + manifest.json
"""

from __future__ import annotations

__version__ = "0.1.0"
