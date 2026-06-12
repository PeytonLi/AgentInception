"""P3 task 6: popup chaos run - bank vs no-bank contrast.

With the popup bank, the model (steered, NOT the loop) returns ``dismiss_modal``;
the loop executes it, the modal's element disappears, and the task resumes. The
control models a no-bank run where the model never emits ``dismiss_modal`` and
the modal therefore persists - the contrast that is a demo talking point.

The loop is deliberately not popup-aware; these tests assert that property by
checking the loop only ever *executes* whatever action the engine returns.
"""

from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.metrics import Metrics
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver

POPUP_URL = "http://localhost:8080/popup.html"
RESUME_URL = "https://news.ycombinator.com/item?id=44210000"


class _ScriptedClient:
    """Minimal engine double: returns scripted actions per call, in order."""

    def __init__(self, script: list[dict]) -> None:
        self._script = script
        self._i = 0
        self.bank_found = True

    async def step(self, payload: dict) -> dict:
        action = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        bank = self.bank_found and payload["page_key"] == "popup:demo"
        return {
            "action": action,
            "bank_found": bank,
            "injected_layers": [8, 12, 16, 20] if bank else [],
            "num_slots": 180 if bank else None,
            "visible_tokens": 150 if payload["mode"] == "mi" else 400,
            "baseline_tokens": 400,
        }


def _runner(client, page, mode="mi"):
    return AgentRunner(
        page=page,
        client=client,
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="Dismiss the cookie modal and continue.",
        session_id=f"popup-{mode}",
        mode=mode,
        metrics=Metrics(),
        num_slots_by_page={"popup:demo": 180},
    )


async def test_popup_bank_dismisses_modal_and_resumes():
    # Steered run: dismiss_modal -> goto (resume) -> done.
    client = _ScriptedClient(
        [
            {"action": "dismiss_modal", "selector": "#accept-cookies"},
            {"action": "goto", "url": RESUME_URL},
            {"action": "done", "result": {"ok": True}},
        ]
    )
    page = FakePageDriver(
        {POPUP_URL: "<div data-testid='cookie-modal'>cookies</div>", RESUME_URL: "ok"},
        POPUP_URL,
        link_map={},
        modal_urls={POPUP_URL},
    )
    outcome = await _runner(client, page).run(POPUP_URL)

    assert outcome.completed is True
    assert ("dismiss_modal", "#accept-cookies") in page.actions
    assert POPUP_URL in page.dismissed  # modal element gone
    assert any(a[0] == "goto" and a[1] == RESUME_URL for a in page.actions)
    # The first step injected the popup bank.
    assert outcome.transcript[0]["bank_found"] is True
    assert outcome.transcript[0]["injected_layers"] == [8, 12, 16, 20]


async def test_no_bank_control_leaves_modal_in_place():
    # Control: without the popup bank the model just gives up (done) and the
    # modal is never dismissed - the contrast we narrate during the demo.
    client = _ScriptedClient([{"action": "done", "result": {"gave_up": True}}])
    client.bank_found = False
    page = FakePageDriver(
        {POPUP_URL: "<div data-testid='cookie-modal'>cookies</div>"},
        POPUP_URL,
        modal_urls={POPUP_URL},
    )
    outcome = await _runner(client, page, mode="baseline").run(POPUP_URL)

    assert outcome.completed is True
    assert page.dismissed == set()  # modal still present
    assert all(a[0] != "dismiss_modal" for a in page.actions)
    assert outcome.transcript[0]["bank_found"] is False
