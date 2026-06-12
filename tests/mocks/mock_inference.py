"""Mock inference engine. CONTRACTS.md s6/s7, owned by A3 (reused by A4 + C1).

A dependency-light FastAPI stub: scripted Action JSON per page_key, realistic
token accounting, a live WS event feed, and a frame rebroadcast endpoint. It
lets the agent-runner, the web-console, and the integration suite all develop
before the real engine exists.

Run standalone:  uvicorn mock_inference:app --port 8000
Or in-process:   httpx.ASGITransport(app=mock_inference.app)
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

SELECTED_LAYERS = [8, 12, 16, 20]
PROMPT_OVERHEAD = 120  # system + scaffolding tokens
MI_VISIBLE_CAP = 480  # keep mi visible_tokens < 500 by construction

# Scripted answers keyed by page_key. The popup answer is what proves the demo
# claim: the model (not the runner) chooses to dismiss the modal.
_SCRIPT: dict[str, dict[str, Any]] = {
    "hn:front": {
        "action": "goto",
        "url": "https://news.ycombinator.com/item?id=44210000",
    },
    "hn:item": {
        "action": "extract",
        "result": {
            "score": 312,
            "top_commenters": ["pg", "dang", "patio11"],
        },
    },
    "popup:demo": {"action": "dismiss_modal", "selector": "#accept-cookies"},
}
_MALFORMED = "I think we should click the link"  # not valid JSON


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _approx(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


class StepRequest(BaseModel):
    session_id: str
    mode: str
    task: str = ""
    url: str = ""
    page_key: str = "unknown"
    dom_text: str | None = None
    dom_token_count: int = 0
    history: list[str] = []
    step: int = 0
    cum_visible: int = 0
    cum_baseline: int = 0
    reprompt: bool = False


class FrameRequest(BaseModel):
    jpeg_base64: str


def create_app() -> FastAPI:
    app = FastAPI(title="GhostBrowser mock inference")
    app.state.subscribers = set()
    app.state.calls = {}  # session_id -> call count

    async def broadcast(event: dict[str, Any]) -> None:
        event.setdefault("ts", _now())
        for queue in list(app.state.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - bounded drop
                pass

    def decide(req: StepRequest) -> Any:
        sid = req.session_id
        if sid.startswith("malformed-twice"):
            return _MALFORMED
        if sid.startswith("malformed-once"):
            return _SCRIPT["hn:item"] if req.reprompt else _MALFORMED
        if req.page_key in _SCRIPT:
            return _SCRIPT[req.page_key]
        # Unknown page: graceful fallback, just finish.
        return {"action": "done", "result": {"note": "no bank for page"}}

    def token_counts(req: StepRequest) -> tuple[int, int]:
        baseline = req.dom_token_count + PROMPT_OVERHEAD
        if req.mode == "baseline":
            return baseline, baseline
        visible = PROMPT_OVERHEAD + _approx(req.task)
        visible += sum(_approx(h) for h in req.history)
        return min(visible, MI_VISIBLE_CAP), baseline

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "banks_loaded": ["hn:front", "hn:item", "popup:demo"],
        }

    @app.post("/api/v1/step")
    async def step(req: StepRequest) -> dict[str, Any]:
        app.state.calls[req.session_id] = (
            app.state.calls.get(req.session_id, 0) + 1
        )
        action = decide(req)
        visible, baseline = token_counts(req)
        bank_found = req.mode == "mi" and req.page_key in _SCRIPT
        injected = SELECTED_LAYERS if bank_found else []

        await broadcast(
            {
                "type": "layer_injection",
                "layers": injected,
                "active": bool(bank_found),
                "page_key": req.page_key,
                "num_slots": 312 if bank_found else 0,
            }
        )
        cum_visible = req.cum_visible + visible
        cum_baseline = req.cum_baseline + baseline
        await broadcast(
            {
                "type": "token_metrics",
                "session_id": req.session_id,
                "step": req.step,
                "mode": req.mode,
                "visible_tokens": visible,
                "baseline_tokens": baseline,
                "cum_visible": cum_visible,
                "cum_baseline": cum_baseline,
                "kv_savings_ratio": (
                    round(cum_baseline / cum_visible, 1) if cum_visible else 0.0
                ),
            }
        )
        if isinstance(action, dict):
            await broadcast({"type": "action", "step": req.step, "action": action})

        return {
            "action": action,
            "bank_found": bank_found,
            "injected_layers": injected,
            "visible_tokens": visible,
            "baseline_tokens": baseline,
        }

    @app.post("/internal/frame")
    async def internal_frame(frame: FrameRequest) -> dict[str, str]:
        await broadcast(
            {"type": "viewport_frame", "jpeg_base64": frame.jpeg_base64}
        )
        return {"status": "ok"}

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        app.state.subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.subscribers.discard(queue)

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
