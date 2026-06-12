"""agent_steps logging. CONTRACTS.md s5.

Wraps ``ghost_shared.storage.log_step``. Always keeps an in-memory copy of each
row (used by tests and end-of-run summaries) and best-effort writes to
ClickHouse when a client is available. A missing/unreachable ClickHouse
degrades to in-memory only - it must never crash a demo run.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent_runner.steplog")


class StepLogger:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self.rows: list[dict[str, Any]] = []

    @classmethod
    def connect(cls, url: str | None = None, enabled: bool = True) -> "StepLogger":
        if not enabled:
            return cls(None)
        try:
            from ghost_shared import storage

            client = storage.get_client(url)
            client.command("SELECT 1")
            logger.info("ClickHouse step logging enabled")
            return cls(client)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning(
                "ClickHouse unavailable (%s); logging steps in-memory only", exc
            )
            return cls(None)

    def log(
        self,
        *,
        session_id: str,
        step: int,
        mode: str,
        url: str,
        page_key: str,
        action_json: str,
        visible_tokens: int,
        baseline_tokens: int,
        bank_found: bool,
    ) -> None:
        row = {
            "session_id": session_id,
            "step": step,
            "mode": mode,
            "url": url,
            "page_key": page_key,
            "action_json": action_json,
            "visible_tokens": visible_tokens,
            "baseline_tokens": baseline_tokens,
            "bank_found": bank_found,
        }
        self.rows.append(row)
        if self._client is not None:
            try:
                from ghost_shared import storage

                storage.log_step(self._client, **row)
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning("failed to write step %s to ClickHouse: %s", step, exc)
