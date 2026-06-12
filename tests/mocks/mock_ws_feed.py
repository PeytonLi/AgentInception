"""Mock WS feed for the web-console (CONTRACTS.md §7).

Replays a canned sequence of events on a loop at ws://localhost:8000/ws/events
so A4 can build the dashboard with zero real backend. Hand to A3 later.

    pip install websockets
    python tests/mocks/mock_ws_feed.py
"""

import asyncio
import base64
import json
from datetime import datetime, timezone

import websockets

# 1x1 transparent-ish JPEG placeholder so the <img> renders something.
_TINY_JPEG = base64.b64encode(
    bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "07090908"
        "0a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c"
        "30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc40014"
        "0001000000000000000000000000000000000affc4001401010000000000000000000"
        "0000000000000ffda0008010100003f00d2cf20ffd9"
    )
).decode()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def handler(ws):
    cum_b = cum_v = 0
    step = 0
    while True:
        for mode, key, layers, active, slots, base_tok, vis_tok in [
            ("baseline", "hn:front", [], False, 0, 1400, 1400),
            ("mi", "hn:front", [8, 12, 16, 20], True, 312, 1600, 210),
            ("mi", "hn:item", [8, 12, 16, 20], True, 240, 1800, 212),
            ("mi", "unknown", [], False, 0, 900, 900),
            ("mi", "popup:demo", [8, 12, 16, 20], True, 96, 1300, 198),
        ]:
            step += 1
            cum_b += base_tok
            cum_v += vis_tok
            ratio = round(cum_b / max(cum_v, 1), 1)

            await ws.send(
                json.dumps(
                    {
                        "type": "layer_injection",
                        "ts": now(),
                        "layers": layers,
                        "active": active,
                        "page_key": key,
                        "num_slots": slots,
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "viewport_frame",
                        "ts": now(),
                        "jpeg_base64": _TINY_JPEG,
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "action",
                        "ts": now(),
                        "step": step,
                        "action": {"action": "click", "selector": "a.morelink"},
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "token_metrics",
                        "ts": now(),
                        "session_id": "mock-session",
                        "step": step,
                        "mode": mode,
                        "visible_tokens": vis_tok,
                        "baseline_tokens": base_tok,
                        "cum_visible": cum_v,
                        "cum_baseline": cum_b,
                        "kv_savings_ratio": ratio,
                    }
                )
            )
            msg = (
                f"Bank {key} injected at layers {layers}"
                if active
                else f"No bank for {key}; plain-prompt fallback"
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "log",
                        "ts": now(),
                        "level": "info" if active else "warn",
                        "message": msg,
                    }
                )
            )
            await asyncio.sleep(1.2)


async def main():
    async with websockets.serve(handler, "localhost", 8000):
        print("mock_ws_feed -> ws://localhost:8000/ws/events")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
