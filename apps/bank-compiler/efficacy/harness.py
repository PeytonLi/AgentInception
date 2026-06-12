"""Steering-efficacy harness for KV banks.

Measures whether a bank makes the model behave as if it saw the full DOM
by comparing logit distributions across three conditions:

    1. **Bank-injected** — latent KV bank active, no DOM in prompt
    2. **Full-DOM** — no bank, full DOM text in prompt (ground truth)
    3. **No-context** — no bank, no DOM (baseline ignorance)

A good bank's logit distribution tracks the full-DOM distribution and
diverges from no-context. We measure this with KL divergence:

    - KL-to-DOM  (lower = better): bank matches full-DOM behavior
    - KL-to-empty (higher = better): bank adds real information

The roll-up score averages across probes per page type.

GPU required: this module runs forward passes through Llama-3.1-8B.
Mark tests with ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("efficacy.harness")


# --------------------------------------------------------------------------
# KL divergence
# --------------------------------------------------------------------------
def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) over the top-k token logits, with log-sum-exp stability.

    Both inputs are raw logits (unnormalized). We softmax them first.
    Returns the KL divergence in nats.
    """
    # Numerical stability: subtract max before exp.
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    p = _softmax(p_logits.astype(np.float64))
    q = _softmax(q_logits.astype(np.float64))

    # Clip to avoid log(0).
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------------
# Probe loading
# --------------------------------------------------------------------------
PROBES_DIR = Path(__file__).parent / "probes"


def load_probes(page_key: str) -> list[dict[str, str]]:
    """Load probe prompts for a page type from the probes/ directory.

    Each probe is a dict with at least ``"prompt"`` and ``"description"``.
    """
    slug = page_key.replace(":", "_")
    path = PROBES_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"No probe set at {path}")
    with open(path, "r", encoding="utf-8") as f:
        probes = json.load(f)
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"Probe set at {path} must be a non-empty list")
    return probes


# --------------------------------------------------------------------------
# Single-probe measurement
# --------------------------------------------------------------------------
def measure_probe(
    *,
    model: Any,
    tokenizer: Any,
    probe_prompt: str,
    bank_layers: dict | None = None,
    dom_text: str | None = None,
    top_k: int = 100,
) -> dict[str, np.ndarray]:
    """Run the model on a probe prompt under different conditions.

    Returns a dict with keys ``"bank"``, ``"dom"``, ``"empty"`` each
    containing the top-k logits of the next token after the probe.

    ``bank_layers`` is the dict[int, tuple[Tensor, Tensor]] to inject.
    ``dom_text`` is the full DOM text for the ground-truth condition.
    """
    import torch

    device = next(model.parameters()).device
    results: dict[str, np.ndarray] = {}

    def _get_logits(messages: list[dict], apply_bank: bool = False) -> np.ndarray:
        """Forward pass → top-k logits of the last token."""
        if apply_bank and bank_layers:
            # Import the MI attention machinery.
            from inference_engine.mi_attention import set_banks, clear_banks
            set_banks(model, bank_layers)

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False, return_dict=True)

        logits = out.logits[0, -1].float().cpu().numpy()

        if apply_bank and bank_layers:
            clear_banks(model)

        # Return top-k logits.
        top_indices = np.argsort(logits)[-top_k:]
        return logits[top_indices]

    base_messages = [
        {"role": "system", "content": "You are a web agent analyzing a page."},
        {"role": "user", "content": probe_prompt},
    ]

    # Condition 1: No context (baseline ignorance).
    results["empty"] = _get_logits(base_messages)

    # Condition 2: Full DOM in context (ground truth).
    if dom_text:
        dom_messages = [
            {"role": "system", "content": "You are a web agent analyzing a page."},
            {"role": "user", "content": f"Page content:\n{dom_text[:8000]}\n\n{probe_prompt}"},
        ]
        results["dom"] = _get_logits(dom_messages)

    # Condition 3: Bank injected.
    if bank_layers:
        results["bank"] = _get_logits(base_messages, apply_bank=True)

    return results


# --------------------------------------------------------------------------
# Full bank scoring
# --------------------------------------------------------------------------
def score_bank(
    *,
    model: Any,
    tokenizer: Any,
    bank_layers: dict,
    probes: list[dict[str, str]],
    dom_text: str,
    top_k: int = 100,
) -> dict[str, Any]:
    """Score a bank against all probes. Returns per-probe and roll-up scores.

    Result dict:
        {
            "per_probe": [
                {"description": ..., "kl_to_dom": ..., "kl_to_empty": ...},
                ...
            ],
            "avg_kl_to_dom": float,
            "avg_kl_to_empty": float,
            "rollup_score": float,   # kl_to_empty - kl_to_dom (higher = better)
        }
    """
    per_probe: list[dict[str, Any]] = []

    for probe in probes:
        prompt = probe["prompt"]
        desc = probe.get("description", prompt[:60])

        results = measure_probe(
            model=model,
            tokenizer=tokenizer,
            probe_prompt=prompt,
            bank_layers=bank_layers,
            dom_text=dom_text,
            top_k=top_k,
        )

        kl_to_dom = float("nan")
        kl_to_empty = float("nan")

        if "bank" in results and "dom" in results:
            kl_to_dom = kl_divergence(results["bank"], results["dom"])
        if "bank" in results and "empty" in results:
            kl_to_empty = kl_divergence(results["bank"], results["empty"])

        per_probe.append({
            "description": desc,
            "kl_to_dom": kl_to_dom,
            "kl_to_empty": kl_to_empty,
        })
        logger.info(
            "  probe %r: KL-to-DOM=%.4f  KL-to-empty=%.4f",
            desc[:40], kl_to_dom, kl_to_empty,
        )

    # Averages (ignoring NaN).
    kl_doms = [p["kl_to_dom"] for p in per_probe if not np.isnan(p["kl_to_dom"])]
    kl_empties = [p["kl_to_empty"] for p in per_probe if not np.isnan(p["kl_to_empty"])]

    avg_kl_to_dom = float(np.mean(kl_doms)) if kl_doms else float("nan")
    avg_kl_to_empty = float(np.mean(kl_empties)) if kl_empties else float("nan")
    rollup = avg_kl_to_empty - avg_kl_to_dom if not (np.isnan(avg_kl_to_dom) or np.isnan(avg_kl_to_empty)) else float("nan")

    return {
        "per_probe": per_probe,
        "avg_kl_to_dom": avg_kl_to_dom,
        "avg_kl_to_empty": avg_kl_to_empty,
        "rollup_score": rollup,
    }
