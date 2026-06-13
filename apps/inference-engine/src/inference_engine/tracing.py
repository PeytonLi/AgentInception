"""Langfuse tracing for inference engine steps (CONTRACTS §6–§7).

Thin wrapper around the langfuse SDK that is a **no-op** when no API keys are
configured, matching the graceful-degradation pattern used for ClickHouse
(``steplog.py``). A missing Langfuse instance must never crash a demo run.

Design
------

Each ``/api/v1/step`` call creates one **trace** (``name="agent_step"``) with:

* Trace-level metadata: session_id, mode, step, page_key, url, task
* ``bank_injection`` span — whether a bank was found & injected, layer count
* ``llm_generate`` span — prompt tokens, completion tokens, latency, raw action
* Trace scores — visible_tokens, baseline_tokens, kv_savings_ratio, bank_found

The tracer also batches generations within a retry under the same trace so you
can see whether the model needed a second shot at producing valid Action JSON.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import Settings

logger = logging.getLogger("inference_engine.tracing")


class StepTracer:
    """Per-step trace handle. Created once per ``/api/v1/step`` call.

    If Langfuse is not configured, all methods are safe no-ops that never raise.
    """

    def __init__(self, langfuse_client: Any | None, trace: Any | None) -> None:
        self._client = langfuse_client
        self._trace = trace
        self._gen_span: Any | None = None

    # ------------------------------------------------------------------
    # Public API (called from service.py)
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._trace is not None

    def span_bank_injection(
        self,
        *,
        bank_found: bool,
        page_key: str,
        injected_layers: list[int],
        num_slots: int,
    ) -> None:
        """Record the bank lookup + injection as a child span."""
        if not self.active:
            return
        span = self._trace.span(
            name="bank_injection",
            input={"page_key": page_key},
            output={
                "bank_found": bank_found,
                "injected_layers": injected_layers,
                "num_slots": num_slots,
            },
        )
        span.end()

    def gen_span_start(
        self,
        *,
        attempt: int,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> float:
        """Begin an ``llm_generate`` span. Returns the start timestamp."""
        if not self.active:
            return 0.0
        self._gen_span = self._trace.span(
            name="llm_generate",
            input={
                "attempt": attempt,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
            },
        )
        return time.perf_counter()

    def gen_span_end(
        self,
        *,
        start_s: float,
        raw_output: str,
        completion_tokens: int,
        parsed_ok: bool,
    ) -> None:
        """Close the ``llm_generate`` span with output metadata."""
        if not self.active or self._gen_span is None:
            return
        latency_s = round(time.perf_counter() - start_s, 3)
        self._gen_span.end(
            output={
                "raw_length": len(raw_output),
                "completion_tokens": completion_tokens,
                "parsed_ok": parsed_ok,
                "latency_s": latency_s,
            }
        )
        self._gen_span = None

    def set_scores(
        self,
        *,
        visible_tokens: int,
        baseline_tokens: int,
        kv_savings_ratio: float,
        bank_found: bool,
        injected_layers: int,
        latency_s: float,
    ) -> None:
        """Attach numeric scores to the trace for dashboard comparisons."""
        if not self.active:
            return
        self._trace.score(name="visible_tokens", value=float(visible_tokens))
        self._trace.score(name="baseline_tokens", value=float(baseline_tokens))
        self._trace.score(name="kv_savings_ratio", value=float(kv_savings_ratio))
        self._trace.score(name="bank_found", value=1.0 if bank_found else 0.0)
        self._trace.score(name="injected_layers", value=float(injected_layers))
        self._trace.score(name="latency_s", value=latency_s)

    def update_output(self, action: dict) -> None:
        """Set the trace-level output to the final action."""
        if not self.active:
            return
        self._trace.update(output={"action": action})

    def update_error(self, detail: str) -> None:
        """Mark the trace as errored with a detail message."""
        if not self.active:
            return
        self._trace.update(level="ERROR", status_message=detail[:500])


class TraceFactory:
    """Factory that creates ``StepTracer`` instances (or no-op stubs).

    Holds one ``Langfuse`` client for the process lifetime. Created once at
    server startup and passed into ``StepService``.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = self._build_client(settings)

    @classmethod
    def disabled(cls) -> "TraceFactory":
        """Return a no-op factory (no Langfuse client)."""
        factory = object.__new__(cls)
        factory._client = None
        return factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def step_tracer(
        self,
        *,
        session_id: str,
        mode: str,
        task: str,
        url: str,
        page_key: str,
        step: int,
    ) -> StepTracer:
        """Create a trace for a single ``/api/v1/step`` request."""
        if self._client is None:
            return StepTracer(None, None)
        trace = self._client.trace(
            name="agent_step",
            session_id=session_id,
            input={
                "mode": mode,
                "task": task,
                "url": url,
                "page_key": page_key,
                "step": step,
            },
        )
        return StepTracer(self._client, trace)

    def flush(self) -> None:
        """Best-effort flush of pending spans before process exit."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            logger.debug("Langfuse flush failed (non-critical)", exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(settings: Settings) -> Any | None:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.info(
                "Langfuse tracing disabled (LANGFUSE_PUBLIC_KEY / "
                "LANGFUSE_SECRET_KEY not set)"
            )
            return None
        try:
            from langfuse import Langfuse

            kwargs: dict[str, Any] = {
                "public_key": settings.langfuse_public_key,
                "secret_key": settings.langfuse_secret_key,
            }
            if settings.langfuse_host:
                kwargs["host"] = settings.langfuse_host
            client = Langfuse(**kwargs)
            client.auth_check()
            logger.info(
                "Langfuse tracing enabled (host=%s)", settings.langfuse_host or "cloud"
            )
            return client
        except Exception as exc:
            logger.warning(
                "Langfuse init failed (%s); tracing disabled. The server will "
                "continue without observability.",
                exc,
            )
            return None
