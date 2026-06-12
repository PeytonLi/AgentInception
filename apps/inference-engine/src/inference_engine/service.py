"""Step orchestration: bank lookup -> prompt -> generate -> Action JSON,
with per-session token accounting and WS events (CONTRACTS §6-§7)."""

import asyncio
import logging

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from .bank_registry import BankRegistry
from .config import NUM_LAYERS
from .engine import RETRY_SUFFIX, GenerationBackend, build_messages, parse_action_json
from .schemas import StepRequest, StepResponse
from .ws_hub import EventHub

logger = logging.getLogger("inference_engine.service")


class StepService:
    def __init__(self, backend: GenerationBackend, registry: BankRegistry, hub: EventHub):
        self.backend = backend
        self.registry = registry
        self.hub = hub
        self._sessions: dict[str, dict[str, int]] = {}
        # one request mutates model bank state at a time
        self._model_lock = asyncio.Lock()

    async def step(self, req: StepRequest) -> StepResponse:
        bank = self.registry.get(req.page_key) if req.mode == "mi" else None
        bank_found = bank is not None
        num_slots = self.registry.num_slots(req.page_key) if bank_found else 0

        # bank_found=false in mi mode => silently fall back to dom_text when the
        # runner provided it (§6 graceful-fallback demo moment, not an error)
        include_dom = req.mode == "baseline" or (req.mode == "mi" and not bank_found and bool(req.dom_text))
        messages = build_messages(
            req.task, req.url, req.history,
            dom_text=req.dom_text if include_dom else None,
            latent_context=bank_found,
        )

        async with self._model_lock:
            injected = await run_in_threadpool(self.backend.apply_banks, bank)
            await self.hub.broadcast(
                {
                    "type": "layer_injection",
                    "layers": injected,
                    "active": bank_found,
                    "page_key": req.page_key,
                    "num_slots": num_slots,
                }
            )
            if bank_found:
                await self.hub.broadcast(
                    {
                        "type": "log",
                        "level": "info",
                        "message": f"Bank {req.page_key} injected at layers {injected}",
                    }
                )

            raw = await run_in_threadpool(self.backend.generate, messages)
            try:
                action = parse_action_json(raw)
            except ValueError:
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": RETRY_SUFFIX},
                ]
                raw = await run_in_threadpool(self.backend.generate, retry_messages)
                try:
                    action = parse_action_json(raw)
                except ValueError:
                    logger.error("model produced no valid Action JSON twice: %r", raw[:200])
                    raise HTTPException(
                        status_code=502,
                        detail=f"model produced no valid Action JSON after retry: {raw[:200]!r}",
                    )

        visible_tokens = self.backend.count_prompt_tokens(messages)
        if req.mode == "baseline":
            baseline_tokens = visible_tokens
        else:
            # prompt overhead = the dom-less prompt; what baseline WOULD have sent
            # is that overhead plus the full DOM token count from the runner
            if include_dom:
                overhead_messages = build_messages(
                    req.task, req.url, req.history, dom_text=None, latent_context=bank_found
                )
                overhead = self.backend.count_prompt_tokens(overhead_messages)
            else:
                overhead = visible_tokens
            baseline_tokens = req.dom_token_count + overhead

        session = self._sessions.setdefault(req.session_id, {"cum_visible": 0, "cum_baseline": 0})
        session["cum_visible"] += visible_tokens
        session["cum_baseline"] += baseline_tokens

        # README pitch math: KV_ratio = (NUM_LAYERS * T_guidance) / (L_ctrl * S_bank)
        if bank_found and injected and num_slots and req.dom_token_count:
            kv_savings_ratio = (NUM_LAYERS * req.dom_token_count) / (len(injected) * num_slots)
        else:
            kv_savings_ratio = 1.0

        await self.hub.broadcast({"type": "action", "step": req.step, "action": action})
        await self.hub.broadcast(
            {
                "type": "token_metrics",
                "session_id": req.session_id,
                "step": req.step,
                "mode": req.mode,
                "visible_tokens": visible_tokens,
                "baseline_tokens": baseline_tokens,
                "cum_visible": session["cum_visible"],
                "cum_baseline": session["cum_baseline"],
                "kv_savings_ratio": round(kv_savings_ratio, 1),
            }
        )

        return StepResponse(
            action=action,
            bank_found=bank_found,
            injected_layers=injected,
            visible_tokens=visible_tokens,
            baseline_tokens=baseline_tokens,
        )
