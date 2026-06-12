"""Perf sanity for MI injection (brief: task 5).

Measures per-step (forward-pass) latency WITH vs WITHOUT a bank injected at the
selected layers and reports the injection overhead. The thesis only holds if
the overhead is small and roughly constant, since the bank adds a fixed
num_slots to every attended sequence regardless of prompt length.

Local self-test (no GPU/HF):  python scripts/bench_mi.py --tiny
EC2 with a real bank:         python scripts/bench_mi.py --real \
    --page-key hn:front --iters 30
Append the printed table to docs/handoff/phase-2/notes/p2-mi-validation.md.
"""

import argparse
import statistics
import sys
import time

import torch


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_forward(model, input_ids, iters: int, warmup: int) -> list[float]:
    device = input_ids.device
    for _ in range(warmup):
        model(input_ids)
    _sync(device)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(input_ids)
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarize(name: str, ms: list[float]) -> dict:
    s = sorted(ms)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    row = {
        "name": name,
        "mean": statistics.mean(ms),
        "median": statistics.median(ms),
        "p90": p90,
    }
    print(f"  {name:<14} mean={row['mean']:7.2f}ms  median={row['median']:7.2f}ms  "
          f"p90={row['p90']:7.2f}ms  (n={len(ms)})")
    return row


def build_tiny():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from inference_engine.mi_attention import clear_banks, set_banks, swap_mi_attention

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=503, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    layers = [1, 3]
    swap_mi_attention(model, layers)
    input_ids = torch.randint(3, 500, (1, 64))
    gen = torch.Generator().manual_seed(7)
    bank = {idx: (torch.randn(2, 16, 8, generator=gen),
                  torch.randn(2, 16, 8, generator=gen)) for idx in layers}
    return model, input_ids, bank, clear_banks, set_banks


def build_real(banks_dir, page_key, prompt):
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.config import Settings
    from inference_engine.engine import LlamaBackend
    from inference_engine.mi_attention import clear_banks, set_banks

    settings = Settings.from_env()
    registry = BankRegistry.load(settings.clickhouse_url, banks_dir or settings.banks_dir)
    bank = registry.get(page_key)
    if bank is None:
        sys.exit(f"no bank for page_key={page_key!r}; available: {registry.page_keys}")
    backend = LlamaBackend.load(settings)
    model, tokenizer = backend.model, backend.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"{settings.model_id}, bank {page_key!r} "
          f"({registry.num_slots(page_key)} slots), prompt {input_ids.shape[1]} tokens")
    return model, input_ids, bank, clear_banks, set_banks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tiny", action="store_true", help="tiny model self-test (default)")
    mode.add_argument("--real", action="store_true", help="real Llama-3.1-8B + real bank")
    parser.add_argument("--banks-dir", default=None)
    parser.add_argument("--page-key", default="hn:front")
    parser.add_argument("--prompt", default="The top story on Hacker News today is")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if args.real:
        model, input_ids, bank, clear_banks, set_banks = build_real(
            args.banks_dir, args.page_key, args.prompt
        )
    else:
        model, input_ids, bank, clear_banks, set_banks = build_tiny()

    device = input_ids.device
    print(f"device={device}  iters={args.iters}  warmup={args.warmup}\n")

    clear_banks(model)
    base = summarize("no bank", time_forward(model, input_ids, args.iters, args.warmup))

    set_banks(model, bank)
    mi = summarize("with bank", time_forward(model, input_ids, args.iters, args.warmup))
    clear_banks(model)

    overhead = mi["median"] - base["median"]
    pct = 100.0 * overhead / base["median"] if base["median"] else float("nan")
    print(f"\n  injection overhead (median): {overhead:+.2f}ms  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
