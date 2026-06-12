"""H+4 shape-sync artifact (brief: definition of done).

Prints the same prompt's top-5 next tokens WITH and WITHOUT a bank injected —
visibly different output proves the bank reaches the logits.

Local (no GPU / no HF access):    python scripts/prove_injection.py --tiny
EC2 with a compiled/fixture bank: python scripts/prove_injection.py \
    --banks-dir ../../banks --page-key hn:front --prompt "Top story on Hacker News:"
"""

import argparse
import sys

import torch


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = p_logits.softmax(-1)
    return (p * (p_logits.log_softmax(-1) - q_logits.log_softmax(-1))).sum().item()


def show(label: str, logits: torch.Tensor, decode) -> None:
    probs = logits.softmax(-1)
    top = probs.topk(5)
    print(f"\n  {label}")
    for rank, (p, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"    {rank}. {decode(idx):<24} p={p:.4f}")


@torch.no_grad()
def run_tiny() -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny random-weight Llama (same GQA architecture) + random bank."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    from inference_engine.mi_attention import swap_mi_attention

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=503, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    layers = [1, 3]
    swap_mi_attention(model, layers)

    torch.manual_seed(123)
    input_ids = torch.randint(3, 500, (1, 12))

    base_logits = model(input_ids).logits[0, -1]

    gen = torch.Generator().manual_seed(7)
    for idx in layers:
        model.model.layers[idx].self_attn.set_bank(
            torch.randn(2, 16, 8, generator=gen), torch.randn(2, 16, 8, generator=gen)
        )
    banked_logits = model(input_ids).logits[0, -1]

    print(f"tiny model, random bank at layers {layers}, prompt = 12 random token ids")
    show("WITHOUT bank", base_logits, lambda i: f"<token {i}>")
    show("WITH bank", banked_logits, lambda i: f"<token {i}>")
    return base_logits, banked_logits


@torch.no_grad()
def run_real(banks_dir: str, page_key: str, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
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

    clear_banks(model)
    base_logits = model(input_ids).logits[0, -1].float()

    injected = set_banks(model, bank)
    banked_logits = model(input_ids).logits[0, -1].float()

    print(f"{settings.model_id}, bank {page_key!r} ({registry.num_slots(page_key)} slots) "
          f"injected at layers {injected}")
    print(f"prompt: {prompt!r}")
    decode = lambda i: repr(tokenizer.decode([i]))
    show("WITHOUT bank", base_logits, decode)
    show("WITH bank", banked_logits, decode)
    return base_logits, banked_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true", help="tiny random model, runs anywhere")
    parser.add_argument("--banks-dir", default=None)
    parser.add_argument("--page-key", default="hn:front")
    parser.add_argument("--prompt", default="The top story on Hacker News today is")
    args = parser.parse_args()

    if args.tiny:
        base, banked = run_tiny()
    else:
        base, banked = run_real(args.banks_dir, args.page_key, args.prompt)

    kl = kl_divergence(base, banked)
    print(f"\n  KL(base || banked) = {kl:.4f}  ({'INJECTION VISIBLE' if kl > 1e-3 else 'NO EFFECT — investigate!'})")


if __name__ == "__main__":
    main()
