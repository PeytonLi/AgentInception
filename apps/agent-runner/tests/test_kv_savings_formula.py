"""P3 task 3: the structural KV-cache ratio matches the README formula exactly.

The runner prints two numbers and they must both be honest:
  * observed   = cum_baseline / cum_visible   (grows over a run)
  * structural = (NUM_LAYERS * T_guidance) / (L_injected * S_bank)  (per page)

These tests pin the formula and prove the runner derives the structural ratio
from real tracked inputs (DOM token count, injected layer count, bank slots) -
never a hard-coded display value.
"""

from __future__ import annotations

import pytest

from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.metrics import Metrics, kv_cache_ratio
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver


def test_readme_headline_number():
    # README: (32 * 14000) / (4 * 312) ~= 358.97 -> 359.0
    assert kv_cache_ratio(14000, 312, 4) == pytest.approx(359.0, abs=0.1)


def test_ratio_zero_without_a_bank():
    assert kv_cache_ratio(14000, 0, 0) == 0.0
    assert kv_cache_ratio(0, 312, 4) == 0.0


def test_metrics_structural_ratio_uses_paired_inputs():
    m = Metrics()
    # A small page (front) then the densest page (item); the headline ratio must
    # pair T_guidance and S_bank from the SAME (densest) step.
    m.record(150, 1016, dom_token_count=896, num_slots=312, num_injected_layers=4)
    m.record(150, 13285, dom_token_count=13165, num_slots=420, num_injected_layers=4)
    assert m.peak_dom_tokens == 13165
    assert m.bank_slots == 420
    assert m.injected_layers == 4
    assert m.structural_kv_ratio == kv_cache_ratio(13165, 420, 4)


def test_baseline_records_no_structural_inputs():
    # Baseline mode never injects a bank: structural ratio stays 0 (honest).
    m = Metrics()
    m.record(14000, 14000)  # no num_slots/injected -> not a bank step
    assert m.structural_kv_ratio == 0.0
    assert m.kv_savings_ratio == 1.0


@pytest.mark.asyncio
async def test_loop_resolves_num_slots_from_manifest(pages, inference_client):
    """The mock omits num_slots; the runner fills it from the manifest map."""
    client, _app = inference_client
    runner = AgentRunner(
        page=FakePageDriver(pages, "https://news.ycombinator.com/"),
        client=client,
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="Find the top AI story and extract score + top 3 commenters.",
        session_id="kv-mi-001",
        mode="mi",
        metrics=Metrics(),
        num_slots_by_page={"hn:front": 312, "hn:item": 420},
    )
    outcome = await runner.run("https://news.ycombinator.com/")
    assert outcome.completed
    # A bank-backed step populated the structural inputs from the manifest.
    assert outcome.metrics.bank_slots in (312, 420)
    assert outcome.metrics.injected_layers == 4
    assert outcome.metrics.structural_kv_ratio > 0
    # The transcript carries the per-step num_slots the chart/PR will cite.
    assert any(s["num_slots"] in (312, 420) for s in outcome.transcript)
