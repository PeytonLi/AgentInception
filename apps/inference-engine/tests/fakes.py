"""Test doubles for the server tests (brief: step endpoint tests run with the model mocked)."""

import torch

from inference_engine.bank_registry import BankRegistry


class FakeBackend:
    """Scripted GenerationBackend: counts tokens by whitespace, records prompts."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ['{"action": "click", "selector": "a.morelink"}']
        self.prompts: list[list[dict]] = []
        self.applied_banks: list[object] = []
        self.model_loaded = True

    def generate(self, messages: list[dict]) -> str:
        self.prompts.append(messages)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return len(self._flat(messages).split())

    def apply_banks(self, layer_banks) -> list[int]:
        self.applied_banks.append(layer_banks)
        return sorted(layer_banks.keys()) if layer_banks else []

    @staticmethod
    def _flat(messages: list[dict]) -> str:
        return "\n".join(m["content"] for m in messages)

    def last_prompt_text(self) -> str:
        return self._flat(self.prompts[-1])


def make_test_registry(page_keys=("hn:front",), num_slots=16) -> BankRegistry:
    gen = torch.Generator().manual_seed(3)
    banks = {
        pk: {
            layer: (
                torch.randn(8, num_slots, 128, generator=gen),
                torch.randn(8, num_slots, 128, generator=gen),
            )
            for layer in (8, 12, 16, 20)
        }
        for pk in page_keys
    }
    return BankRegistry(banks, source="test")
