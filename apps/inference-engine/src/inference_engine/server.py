"""FastAPI app — CONTRACTS §6/§7/§9. Port 8000.

create_app(backend=..., registry=...) injects test doubles; with no arguments
the real Llama backend and bank registry are loaded at startup (EC2 path).
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .bank_registry import BankRegistry
from .config import Settings
from .engine import GenerationBackend
from .schemas import FramePayload, StepRequest, StepResponse
from .service import StepService
from .tracing import TraceFactory
from .ws_hub import EventHub

logger = logging.getLogger("inference_engine.server")


def create_app(
    backend: Optional[GenerationBackend] = None,
    registry: Optional[BankRegistry] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    hub = EventHub()
    trace_factory = TraceFactory(settings)
    state: dict = {"service": None, "registry": registry, "backend": backend}

    if backend is not None and registry is not None:
        state["service"] = StepService(backend, registry, hub, trace_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["service"] is None:
            from .engine import LlamaBackend

            logger.info("startup: loading banks then model (this takes a while)")
            state["registry"] = await run_in_threadpool(
                BankRegistry.load, settings.clickhouse_url, settings.banks_dir
            )
            state["backend"] = await run_in_threadpool(LlamaBackend.load, settings)
            state["service"] = StepService(state["backend"], state["registry"], hub, trace_factory)
            logger.info("startup complete: banks=%s", state["registry"].page_keys)
        yield

    app = FastAPI(title="AgentInception inference engine", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        service: Optional[StepService] = state["service"]
        return {
            "status": "ok",
            "model_loaded": bool(service and service.backend.model_loaded),
            "banks_loaded": state["registry"].page_keys if state["registry"] else [],
        }

    @app.post("/api/v1/step", response_model=StepResponse)
    async def step(req: StepRequest) -> StepResponse:
        return await state["service"].step(req)

    @app.post("/internal/frame")
    async def frame(payload: FramePayload) -> dict:
        # pushed by agent-runner; engine just rebroadcasts on the WS (§7)
        await hub.broadcast(
            {"type": "viewport_frame", "jpeg_base64": payload.jpeg_base64}
        )
        return {"ok": True}

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; we never expect client messages
        except WebSocketDisconnect:
            hub.disconnect(ws)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    uvicorn.run(create_app(settings=settings), host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
