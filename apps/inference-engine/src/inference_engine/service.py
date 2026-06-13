"""Step orchestration: bank lookup -> prompt -> generate -> Action JSON,
with per-session token accounting and WS events (CONTRACTS §6-§7)."""

import asyncio
import logging
import time

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from .bank_registry import BankRegistry
from .config import GENERATION_MAX_NEW_TOKENS, NUM_LAYERS
from .engine import RETRY_SUFFIX, GenerationBackend, build_messages, parse_action_json
from .schemas import StepRequest, StepResponse
from .tracing import StepTracer, TraceFactory
from .ws_hub import EventHub

logger = logging.getLogger("inference_engine.service")


class StepService:
    def __init__(
        self,
        backend: GenerationBackend,
        registry: BankRegistry,
        hub: EventHub,
        trace_factory: TraceFactory | None = None,
    ):
        self.backend = backend
        self.registry = registry
        self.hub = hub
        self.trace_factory = trace_factory or TraceFactory.disabled()
        self._sessions: dict[str, dict[str, int]] = {}
        # one request mutates model bank state at a time
        self._model_lock = asyncio.Lock()

    async def step(self, req: StepRequest) -> StepResponse:
        t0 = time.perf_counter()
        tracer = self.trace_factory.step_tracer(
            session_id=req.session_id,
            mode=req.mode,
            task=req.task,
            url=req.url,
            page_key=req.page_key,
            step=req.step,
        )
        try:
            result = await self._step_impl(req, tracer)
        except HTTPException:
            tracer.update_error(f"HTTP 502 at step {req.step}")
            raise
        except Exception as exc:
            tracer.update_error(str(exc))
            raise
        if result.bank_found and result.injected_layers:
            denom = len(result.injected_layers) * max(
                self.registry.num_slots(req.page_key), 1
            )
            kv_ratio = (
                round((NUM_LAYERS * req.dom_token_count) / denom, 1)
                if denom > 0 and req.dom_token_count > 0
                else 1.0
            )
        else:
            kv_ratio = 1.0
        latency_s = round(time.perf_counter() - t0, 3)
        tracer.set_scores(
            visible_tokens=result.visible_tokens,
            baseline_tokens=result.baseline_tokens,
            kv_savings_ratio=kv_ratio,
            bank_found=result.bank_found,
            injected_layers=len(result.injected_layers),
            latency_s=latency_s,
        )
        tracer.update_output(result.action)
        return result

    async def _step_impl(self, req: StepRequest, tracer: StepTracer) -> StepResponse:
        bank = self.registry.get(req.page_key) if req.mode == "mi" else None
        bank_found = bank is not None
        num_slots = self.registry.num_slots(req.page_key) if bank_found else 0

        # bank_found=false in mi mode => silently fall back to dom_text when the
        # runner provided it (§6 graceful-fallback demo moment, not an error)
        include_dom = req.mode == "baseline" or (
            req.mode == "mi" and not bank_found and bool(req.dom_text)
        )
        messages = build_messages(
            req.task,
            req.url,
            req.history,
            dom_text=req.dom_text if include_dom else None,
            latent_context=bank_found,
        )

        async with self._model_lock:
            injected = await run_in_threadpool(self.backend.apply_banks, bank)
            tracer.span_bank_injection(
                bank_found=bank_found,
                page_key=req.page_key,
                injected_layers=injected,
                num_slots=num_slots,
            )
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

            action, raw_output = await self._generate_with_trace(
                messages, tracer, attempt=1
            )
            if action is None:
                retry_messages = messages + [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": RETRY_SUFFIX},
                ]
                action, raw_output = await self._generate_with_trace(
                    retry_messages, tracer, attempt=2
                )
                if action is None:
                    logger.error(
                        "model produced no valid Action JSON twice: %r",
                        raw_output[:200],
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "model produced no valid Action JSON after retry: "
                            f"{raw_output[:200]!r}"
                        ),
                    )

        visible_tokens = self.backend.count_prompt_tokens(messages)
        if req.mode == "baseline":
            baseline_tokens = visible_tokens
        else:
            # prompt overhead = the dom-less prompt; what baseline WOULD have sent
            # is that overhead plus the full DOM token count from the runner
            if include_dom:
                overhead_messages = build_messages(
                    req.task,
                    req.url,
                    req.history,
                    dom_text=None,
                    latent_context=bank_found,
                )
                overhead = self.backend.count_prompt_tokens(overhead_messages)
            else:
                overhead = visible_tokens
            baseline_tokens = req.dom_token_count + overhead

        session = self._sessions.setdefault(
            req.session_id, {"cum_visible": 0, "cum_baseline": 0}
        )
        session["cum_visible"] += visible_tokens
        session["cum_baseline"] += baseline_tokens

        # README pitch math: KV_ratio = (NUM_LAYERS * T_guidance) / (L_ctrl * S_bank)
        if bank_found and injected and num_slots and req.dom_token_count:
            kv_savings_ratio = (NUM_LAYERS * req.dom_token_count) / (
                len(injected) * num_slots
            )
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

    async def _generate_with_trace(
        self,
        messages: list[dict],
        tracer: StepTracer,
        *,
        attempt: int,
    ) -> tuple[dict | None, str]:
        """Generate + parse Action JSON, with a Langfuse span per attempt."""
        prompt_tokens = self.backend.count_prompt_tokens(messages)
        gen_start = tracer.gen_span_start(
            attempt=attempt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        )
        raw = await run_in_threadpool(self.backend.generate, messages)
        completion_tokens = self.backend.count_tokens(raw)
        try:
            action = parse_action_json(raw)
            parsed_ok = True
        except ValueError:
            action = None
            parsed_ok = False
        tracer.gen_span_end(
            start_s=gen_start,
            raw_output=raw,
            completion_tokens=completion_tokens,
            parsed_ok=parsed_ok,
        )
        return action, raw

