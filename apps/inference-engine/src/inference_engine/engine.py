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

SYSTEM_PROMPT = """You are AgentInception, a web agent driving a real browser to complete the user's task step by step.

RULES:
- NEVER use "extract" or "done" on the first step. Navigate and interact first.
- Only use "extract" when you are ON the page containing the data.
- Only use "done" when you have ACTUALLY collected the answer through real interactions.
- For clicks, use real CSS selectors you can see on the page.

Include a short "thought" explaining your reasoning, then the action. Respond with EXACTLY ONE JSON object and no prose. Allowed actions:
{"thought": "<one sentence reasoning>", "action": "goto", "url": "<absolute url>"}
{"thought": "<one sentence reasoning>", "action": "click", "selector": "<css selector>"}
{"thought": "<one sentence reasoning>", "action": "dismiss_modal", "selector": "<css selector>"}
{"thought": "<one sentence reasoning>", "action": "extract", "result": {<data ACTUALLY extracted from the current page>}}
{"thought": "<one sentence reasoning>", "action": "done", "result": {<the final answer based on REAL interactions>}}"""

RETRY_SUFFIX = "Respond with only the JSON object (include a thought field)."


def cuda_memory_summary() -> str:
    """One-line VRAM footprint for the startup log / P1 metrics baseline."""
    if not torch.cuda.is_available():
        return "cuda: unavailable (CPU run)"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    name = torch.cuda.get_device_name(0)
    return f"cuda[{name}]: {allocated:.1f} GiB allocated, {reserved:.1f} GiB reserved"


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
        import time

        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Llama-3.1-8B is ~16 GB in bf16; bf16 is required to fit an A10G (24 GB).
        # CPU has no bf16 matmul kernels in torch, so fall back to f32 off-GPU.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(
            "loading %s (%s) on %s, attn=%s ...",
            settings.model_id,
            dtype,
            device,
            settings.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id, token=settings.hf_token
        )
        # Llama ships no pad token; generate() and any future batching need one.
        # eos as pad is the standard convention and is masked out of the loss/output.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            attn_implementation=settings.attn_implementation,
            low_cpu_mem_usage=True,
            token=settings.hf_token,
        )
        model.eval()
        swapped = swap_mi_attention(model, SELECTED_LAYERS)
        logger.info("MI attention installed at layers %s", swapped)

        backend = cls(model, tokenizer)
        # Short cold-start smoke: prove the real forward+generate path works and
        # surface dtype/device/OOM failures here, not on the first live request.
        smoke = backend.generate(
            [{"role": "user", "content": "Say OK."}], max_new_tokens=8
        )
        logger.info("smoke generation: %r", smoke[:80])
        logger.info(
            "model ready in %.1fs; %s", time.perf_counter() - t0, cuda_memory_summary()
        )
        return backend

    @property
    def device(self) -> torch.device:
        # With device_map="auto" the input embeddings hold the canonical input
        # device; model.device can be a meta placeholder, so read it from there.
        return self.model.get_input_embeddings().weight.device

    @torch.no_grad()
    def generate(
        self, messages: list[dict], max_new_tokens: Optional[int] = None
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens or GENERATION_MAX_NEW_TOKENS,
            do_sample=False,  # temp 0
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            out[0, input_ids.shape[1] :], skip_special_tokens=True
        )

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        return len(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        )

    def apply_banks(self, layer_banks: Optional[dict]) -> list[int]:
        if not layer_banks:
            clear_banks(self.model)
            return []
        return set_banks(self.model, layer_banks)
