"""WS event hub — CONTRACTS §7. web-console is the only consumer."""

import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket

logger = logging.getLogger("inference_engine.ws")


class EventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: dict) -> None:
        payload = json.dumps({**event, "ts": datetime.now(timezone.utc).isoformat()})
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                logger.info("dropping dead WS client")
                self.disconnect(ws)
