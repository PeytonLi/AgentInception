"""Action JSON parsing + validation. CONTRACTS.md s8.

The model must answer with exactly one JSON object:

    {"action": "goto",         "url": "https://..."}
    {"action": "click",        "selector": "a.morelink"}
    {"action": "dismiss_modal","selector": "#accept-cookies"}
    {"action": "extract",      "result": {...}}
    {"action": "done",         "result": {...}}

The engine returns this under the response ``action`` field. It may arrive as
a parsed object or, if the model emitted raw text, as a string - we handle
both, and raise :class:`ActionParseError` on anything malformed so the loop can
re-prompt once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import ActionParseError

VALID_ACTIONS = {"goto", "click", "dismiss_modal", "extract", "done"}
TERMINAL_ACTIONS = {"extract", "done"}


@dataclass
class Action:
    type: str
    url: str | None = None
    selector: str | None = None
    result: Any = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL_ACTIONS

    def describe(self) -> str:
        if self.type == "goto":
            return f"goto {self.url}"
        if self.type in ("click", "dismiss_modal"):
            return f"{self.type} {self.selector}"
        return self.type


def parse_action(payload: Any) -> Action:
    """Parse and validate an action payload (dict or raw JSON string)."""
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ActionParseError(
                f"action payload is not valid JSON: {payload!r}"
            ) from exc
    elif isinstance(payload, dict):
        data = payload
    else:
        raise ActionParseError(
            f"action payload must be a dict or JSON string, got {type(payload)!r}"
        )

    if not isinstance(data, dict):
        raise ActionParseError(f"action JSON must be an object, got {data!r}")

    atype = data.get("action")
    if atype not in VALID_ACTIONS:
        raise ActionParseError(f"unknown or missing action type: {atype!r}")

    if atype == "goto" and not data.get("url"):
        raise ActionParseError("goto action requires a non-empty 'url'")
    if atype in ("click", "dismiss_modal") and not data.get("selector"):
        raise ActionParseError(f"{atype} action requires a non-empty 'selector'")
    if atype in TERMINAL_ACTIONS and "result" not in data:
        raise ActionParseError(f"{atype} action requires a 'result' field")

    return Action(
        type=atype,
        url=data.get("url"),
        selector=data.get("selector"),
        result=data.get("result"),
        raw=data,
    )
