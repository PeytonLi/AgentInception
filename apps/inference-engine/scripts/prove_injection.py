"""H+4 shape-sync + demo proof artifact (brief: definition of done, task 6).

For one fixed prompt this prints, side by side:
  - the top-k next tokens WITHOUT and WITH a bank injected,
  - KL(base || mi) and KL(mi || base) between the next-token distributions,
  - the tokens whose probability the bank most promotes / suppresses,
  - whether clearing the bank restores a bit-exact baseline.

Visibly different output and KL > 1e-3 are the headline evidence that MI
injection reaches the logits. This is the script the demo runs to prove it.

Local (no GPU / no HF access):  python scripts/prove_injection.py --tiny
EC2 with a real compiled bank:  python scripts/prove_injection.py --real \
    --page-key hn:front --prompt "The top story on Hacker News today is"
"""

import argparse
import sys

import torch


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) in nats."""
    p = p_logits.softmax(-1)
    return (p * (p_logits.log_softmax(-1) - q_logits.log_softmax(-1))).sum().item()


def show(label: str, logits: torch.Tensor, decode, k: int = 5) -> None:
    probs = logits.softmax(-1)
    top = probs.topk(k)
    print(f"\n  {label}")
    for rank, (p, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"    {rank}. {decode(idx):<24} p={p:.4f}")


def show_token_deltas(base: torch.Tensor, banked: torch.Tensor, decode, k: int = 8):
    base_p = base.softmax(-1)
    banked_p = banked.softmax(-1)
    delta = banked_p - base_p
    gain = delta.topk(k)
    loss = (-delta).topk(k)
    print(f"\n  TOP {k} TOKENS THE BANK PROMOTES (dp = p_mi - p_base)")
    for dp, idx in zip(gain.values.tolist(), gain.indices.tolist()):
        print(f"    {decode(idx):<24} dp=+{dp:.4f}  "
              f"({base_p[idx]:.4f} -> {banked_p[idx]:.4f})")
    print(f"\n  TOP {k} TOKENS THE BANK SUPPRESSES")
    for dp, idx in zip(loss.values.tolist(), loss.indices.tolist()):
        print(f"    {decode(idx):<24} dp=-{dp:.4f}  "
              f"({base_p[idx]:.4f} -> {banked_p[idx]:.4f})")


@torch.no_grad()
def run_tiny():
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

    for idx in layers:
        model.model.layers[idx].self_attn.clear_bank()
    restored = model(input_ids).logits[0, -1]
    restore_ok = torch.equal(restored, base_logits)

    print(f"tiny model, random bank at layers {layers}, prompt = 12 random token ids")
    decode = lambda i: f"<token {i}>"
    show("WITHOUT bank", base_logits, decode)
    show("WITH bank", banked_logits, decode)
    return base_logits, banked_logits, decode, restore_ok


@torch.no_grad()
def run_real(banks_dir, page_key: str, prompt: str):
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

    clear_banks(model)
    restored = model(input_ids).logits[0, -1].float()
    restore_ok = torch.equal(restored, base_logits)

    print(f"{settings.model_id}, bank {page_key!r} ({registry.num_slots(page_key)} "
          f"slots) injected at layers {injected}  [source: {registry.source}]")
    print(f"prompt: {prompt!r}")
    decode = lambda i: repr(tokenizer.decode([i]))
    show("WITHOUT bank", base_logits, decode)
    show("WITH bank", banked_logits, decode)
    return base_logits, banked_logits, decode, restore_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tiny", action="store_true",
                      help="tiny random model, runs anywhere (default)")
    mode.add_argument("--real", action="store_true",
                      help="real Llama-3.1-8B + a real compiled bank (needs GPU/HF)")
    parser.add_argument("--banks-dir", default=None)
    parser.add_argument("--page-key", default="hn:front")
    parser.add_argument("--prompt", default="The top story on Hacker News today is")
    parser.add_argument("--topk", type=int, default=8, help="token-delta table size")
    args = parser.parse_args()

    if args.real:
        base, banked, decode, restore_ok = run_real(
            args.banks_dir, args.page_key, args.prompt
        )
    else:
        base, banked, decode, restore_ok = run_tiny()

    show_token_deltas(base, banked, decode, k=args.topk)

    kl_fwd = kl_divergence(base, banked)
    kl_rev = kl_divergence(banked, base)
    verdict = "INJECTION VISIBLE" if kl_fwd > 1e-3 else "NO EFFECT - investigate!"
    print(f"\n  KL(base || mi) = {kl_fwd:.4f}   KL(mi || base) = {kl_rev:.4f}   ({verdict})")
    restore_msg = "PASS (bit-exact)" if restore_ok else "FAIL - clear_bank did not restore baseline"
    print(f"  clear_bank() baseline restore: {restore_msg}")

    if kl_fwd <= 1e-3 or not restore_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
