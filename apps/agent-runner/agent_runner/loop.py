"""The agent driver loop. A3 brief task 3.

One loop serves both modes. Per step it computes the page_key, extracts DOM
text, calls /api/v1/step (baseline sends dom_text; mi sends dom_text=null plus
the dom_token_count baseline WOULD have sent), parses + executes the returned
action, logs the step, and accumulates token metrics. Hard stop at max_steps.

The popup case is deliberately NOT special-cased here: the model (steered by
the popup bank) returns ``dismiss_modal`` and the loop merely executes it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ghost_shared.page_key import page_key as compute_page_key

from .actions import Action, parse_action
from .browser import PageDriver
from .config import MI_VISIBLE_BUDGET, RunnerConfig
from .dom import extract_dom_text
from .errors import ActionExecutionError, ActionParseError
from .inference_client import InferenceClient
from .metrics import Metrics
from .steplog import StepLogger
from .tokenizer import TokenCounter

logger = logging.getLogger("agent_runner.loop")

# Prompt scaffolding overhead (system + formatting) used only as a fallback
# when the engine response omits the authoritative token counts.
_PROMPT_OVERHEAD = 120


@dataclass
class RunResult:
    result: Any
    completed: bool
    steps: int
    metrics: Metrics
    transcript: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner:
    def __init__(
        self,
        *,
        page: PageDriver,
        client: InferenceClient,
        counter: TokenCounter,
        config: RunnerConfig,
        task: str,
        session_id: str,
        mode: str,
        step_logger: StepLogger | None = None,
        metrics: Metrics | None = None,
        num_slots_by_page: dict[str, int] | None = None,
    ) -> None:
        if mode not in ("baseline", "mi"):
            raise ValueError(f"mode must be 'baseline' or 'mi', got {mode!r}")
        self.page = page
        self.client = client
        self.counter = counter
        self.config = config
        self.task = task
        self.session_id = session_id
        self.mode = mode
        self.step_logger = step_logger or StepLogger(None)
        self.metrics = metrics or Metrics()
        self.history: list[str] = []
        # Canonical S_bank per page_key (manifest) for the structural KV ratio
        # when the engine response does not echo num_slots.
        self.num_slots_by_page = num_slots_by_page or {}
        # Per-step record for the recorded demo transcript (P3 brief DoD).
        self.transcript: list[dict[str, Any]] = []

    async def run(self, start_url: str) -> RunResult:
        await self.page.goto(start_url)
        result: Any = None
        completed = False
        step = 0
        for step in range(self.config.max_steps):
            action, resp, page_key = await self._run_step(step)
            if action is None:
                logger.error("step %s aborted: no valid action", step)
                break
            if action.is_terminal:
                result = action.result
                completed = True
                logger.info("task complete at step %s: %s", step, action.type)
                break
            await self._execute(action)
        else:
            logger.warning(
                "hit max_steps=%s without a terminal action", self.config.max_steps
            )
        return RunResult(
            result=result,
            completed=completed,
            steps=self.metrics.steps,
            metrics=self.metrics,
            transcript=list(self.transcript),
        )

    async def _run_step(self, step: int) -> tuple[Action | None, dict[str, Any], str]:
        url = await self.page.url()
        page_key = compute_page_key(url)
        raw_text = await self.page.inner_text()
        dom_truncated, dom_token_count = extract_dom_text(
            raw_text, self.counter, self.config.dom_token_cap
        )
        request = {
            "session_id": self.session_id,
            "mode": self.mode,
            "task": self.task,
            "url": url,
            "page_key": page_key,
            "dom_text": dom_truncated if self.mode == "baseline" else None,
            "dom_token_count": dom_token_count,
            "history": list(self.history),
            "step": step,
            # Extends CONTRACTS s6 per A3 brief task 5 so the engine can emit
            # token_metrics with running cumulative totals.
            "cum_visible": self.metrics.cum_visible,
            "cum_baseline": self.metrics.cum_baseline,
        }

        action, resp = await self._step_with_retry(request)
        if action is None:
            return None, resp, page_key

        visible, baseline = self._resolve_tokens(resp, dom_token_count)
        injected_layers = resp.get("injected_layers") or []
        num_slots = resp.get("num_slots")
        if num_slots is None and resp.get("bank_found"):
            num_slots = self.num_slots_by_page.get(page_key)
        self.metrics.record(
            visible,
            baseline,
            dom_token_count=dom_token_count,
            num_slots=num_slots,
            num_injected_layers=len(injected_layers) if injected_layers else None,
        )
        if self.mode == "mi" and visible >= MI_VISIBLE_BUDGET:
            logger.warning(
                "mi step %s visible_tokens=%s exceeds budget %s",
                step,
                visible,
                MI_VISIBLE_BUDGET,
            )

        self.step_logger.log(
            session_id=self.session_id,
            step=step,
            mode=self.mode,
            url=url,
            page_key=page_key,
            action_json=json.dumps(action.raw),
            visible_tokens=visible,
            baseline_tokens=baseline,
            bank_found=bool(resp.get("bank_found", False)),
        )
        self.transcript.append(
            {
                "step": step,
                "mode": self.mode,
                "url": url,
                "page_key": page_key,
                "action": action.raw,
                "bank_found": bool(resp.get("bank_found", False)),
                "injected_layers": list(injected_layers),
                "num_slots": num_slots,
                "dom_token_count": dom_token_count,
                "visible_tokens": visible,
                "baseline_tokens": baseline,
                "cum_visible": self.metrics.cum_visible,
                "cum_baseline": self.metrics.cum_baseline,
            }
        )
        self.history.append(action.describe())
        return action, resp, page_key

    async def _step_with_retry(
        self, request: dict[str, Any]
    ) -> tuple[Action | None, dict[str, Any]]:
        resp = await self.client.step(request)
        try:
            return parse_action(resp.get("action")), resp
        except ActionParseError as first:
            logger.warning("malformed action, re-prompting once: %s", first)
        retry = dict(request)
        retry["history"] = list(request["history"]) + [
            "Respond with only the JSON object."
        ]
        retry["reprompt"] = True
        resp = await self.client.step(retry)
        try:
            return parse_action(resp.get("action")), resp
        except ActionParseError as second:
            logger.error("action still malformed after retry: %s", second)
            return None, resp

    def _resolve_tokens(
        self, resp: dict[str, Any], dom_token_count: int
    ) -> tuple[int, int]:
        baseline = resp.get("baseline_tokens")
        visible = resp.get("visible_tokens")
        if baseline is None:
            baseline = dom_token_count + _PROMPT_OVERHEAD
        if visible is None:
            if self.mode == "baseline":
                visible = baseline
            else:
                visible = (
                    _PROMPT_OVERHEAD
                    + self.counter.count(self.task)
                    + sum(self.counter.count(h) for h in self.history)
                )
        return int(visible), int(baseline)

    async def _execute(self, action: Action) -> None:
        if action.type == "goto":
            await self.page.goto(action.url or "")
        elif action.type == "click":
            await self._click_with_retry(action.selector or "")
        elif action.type == "dismiss_modal":
            await self.page.dismiss_modal(action.selector or "")
        else:  # pragma: no cover - terminal actions never reach here
            raise ActionExecutionError(f"non-executable action: {action.type}")

    async def _click_with_retry(self, selector: str) -> None:
        try:
            await self.page.click(selector, timeout_ms=5000)
        except Exception as first:
            logger.warning("click %s failed (%s); retrying once", selector, first)
            try:
                await self.page.click(selector, timeout_ms=5000)
            except Exception as second:
                raise ActionExecutionError(
                    f"click {selector!r} failed twice: {second}"
                ) from second
