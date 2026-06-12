"""Project-wide constants. Mirrors CONTRACTS.md §1 exactly.

No agent may diverge from these values. If a constant is wrong, flag it to
the team rather than editing it locally.
"""

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SELECTED_LAYERS = [8, 12, 16, 20]  # 0-indexed decoder layers (model has 32)
NUM_LAYERS = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8  # GQA: each KV head serves 4 query heads
HEAD_DIM = 128
HIDDEN_SIZE = 4096
BANK_DTYPE = "float32"  # serialization dtype (model runs bf16; banks stored f32)
SUMMARY_WORDS = (200, 400)  # Haiku DOM summary target length
HAIKU_MODEL = "claude-haiku-4-5-20251001"
TRANSFORMERS_PIN = "transformers==4.46.*"

# Page keys produced by page_key().
PAGE_KEYS = ("hn:front", "hn:item", "popup:demo", "unknown")
