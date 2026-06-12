"""HTTP client for the inference engine. CONTRACTS.md s6/s7.

Uses ``httpx`` against a ``base_url`` so tests can inject an
``httpx.AsyncClient`` backed by an ASGITransport wrapping the mock app (no real
socket needed). Frame posts are fire-and-forget and never raise into the loop.
"""

from __future__ import annotations

from typing import Any


class InferenceClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http_client: Any | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self._timeout = timeout

    async def __aenter__(self) -> "InferenceClient":
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base, timeout=self._timeout
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _http(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "InferenceClient used outside its async context manager"
            )
        return self._client

    async def healthz(self) -> dict[str, Any]:
        resp = await self._http.get("/healthz")
        resp.raise_for_status()
        return resp.json()

    async def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._http.post("/api/v1/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def post_frame(self, jpeg_base64: str) -> None:
        # Fire-and-forget: the demo must never stall on a dropped frame.
        try:
            await self._http.post(
                "/internal/frame", json={"jpeg_base64": jpeg_base64}
            )
        except Exception:
            pass
