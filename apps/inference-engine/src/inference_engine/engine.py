"""Prompt assembly, Action JSON parsing, and the real Llama backend.

The step service talks to a GenerationBackend protocol so the FastAPI layer
is fully testable with the model mocked (brief: step endpoint tests).
"""

import json
import logging
from typing import Optional, Protocol

import torch

from .config import GENERATION_MAX_NEW_TOKENS, SELECTED_LAYERS, Settings
from .mi_attention import clear_banks, set_banks, swap_mi_attention
from .schemas import ALLOWED_ACTIONS

logger = logging.getLogger("inference_engine.engine")

SYSTEM_PROMPT = """You are GhostBrowser, a web agent driving a real browser to complete the user's task.
Respond with EXACTLY ONE JSON object and no prose. Allowed actions:
{"action": "goto", "url": "<absolute url>"}
{"action": "click", "selector": "<css selector>"}
{"action": "dismiss_modal", "selector": "<css selector>"}
{"action": "extract", "result": {<data extracted from the page>}}
{"action": "done", "result": {<the final answer to the task>}}"""

RETRY_SUFFIX = "Respond with only the JSON object."


class GenerationBackend(Protocol):
    model_loaded: bool

    def generate(self, messages: list[dict]) -> str: ...

    def count_prompt_tokens(self, messages: list[dict]) -> int: ...

    def apply_banks(self, layer_banks: Optional[dict]) -> list[int]: ...


def build_messages(
    task: str,
    url: str,
    history: list[str],
    dom_text: Optional[str],
    latent_context: bool,
) -> list[dict]:
    """Baseline: task + dom_text + history. MI: task + url + history only (tiny)."""
    lines = [f"TASK: {task}", f"CURRENT URL: {url}", "ACTIONS SO FAR:"]
    if history:
        lines += [f"{i + 1}. {entry}" for i, entry in enumerate(history)]
    else:
        lines.append("(none)")
    if latent_context:
        lines.append("(Page context is provided via injected latent memory banks.)")
    if dom_text:
        lines += ["", "PAGE CONTENT (DOM text):", dom_text]
    lines.append("Next action:")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_action_json(text: str) -> dict:
    """Parse the model's Action JSON (§8); tolerates surrounding prose."""
    candidates = [text.strip()]
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("action") in ALLOWED_ACTIONS:
            return parsed
    raise ValueError(f"no valid Action JSON in model output: {text[:200]!r}")


class LlamaBackend:
    """Llama-3.1-8B-Instruct in bfloat16 with MI attention at SELECTED_LAYERS."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_loaded = True

    @classmethod
    def load(cls, settings: Settings) -> "LlamaBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading %s (bfloat16) on %s ...", settings.model_id, device)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=settings.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="sdpa",
            token=settings.hf_token,
        )
        model.eval()
        swapped = swap_mi_attention(model, SELECTED_LAYERS)
        logger.info("MI attention installed at layers %s", swapped)

        backend = cls(model, tokenizer)
        smoke = backend.generate([{"role": "user", "content": "Say OK."}])
        logger.info("smoke generation: %r", smoke[:80])
        return backend

    @torch.no_grad()
    def generate(self, messages: list[dict]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            input_ids,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            do_sample=False,  # temp 0
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))

    def apply_banks(self, layer_banks: Optional[dict]) -> list[int]:
        if not layer_banks:
            clear_banks(self.model)
            return []
        return set_banks(self.model, layer_banks)
