"""CONTRACTS.md §5 — ClickHouse roundtrip. Skipped if no reachable server.

Run `scripts/ch_init.sh` first to exercise this test.
"""

import os

import numpy as np
import pytest

from agentinception_shared import storage
from agentinception_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS


def _client_or_skip():
    try:
        import clickhouse_connect  # noqa: F401
    except ImportError:
        pytest.skip("clickhouse-connect not installed")
    try:
        client = storage.get_client(
            os.environ.get("CLICKHOUSE_URL", storage.DEFAULT_URL)
        )
        client.command("SELECT 1")
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"ClickHouse not reachable: {exc}")
    return client


def _rand_bank(num_slots=16, seed=0):
    rng = np.random.default_rng(seed)
    return {
        layer: (
            rng.standard_normal((NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32),
            rng.standard_normal((NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32),
        )
        for layer in SELECTED_LAYERS
    }


@pytest.mark.integration
def test_clickhouse_bank_roundtrip():
    client = _client_or_skip()
    page_key = "test:roundtrip"
    client.command(
        f"ALTER TABLE {storage.BANKS_TABLE} DELETE WHERE page_key = %(pk)s",
        parameters={"pk": page_key},
    )
    banks = _rand_bank(num_slots=20, seed=99)
    storage.insert_bank(client, page_key, "test.local", banks, "abc123")

    all_banks = storage.load_all_banks(client)
    assert page_key in all_banks
    for layer in SELECTED_LAYERS:
        assert np.array_equal(banks[layer][0], all_banks[page_key][layer][0])
        assert np.array_equal(banks[layer][1], all_banks[page_key][layer][1])

    client.command(
        f"ALTER TABLE {storage.BANKS_TABLE} DELETE WHERE page_key = %(pk)s",
        parameters={"pk": page_key},
    )


@pytest.mark.integration
def test_clickhouse_log_step():
    client = _client_or_skip()
    storage.log_step(
        client,
        session_id="sess-test",
        step=1,
        mode="mi",
        url="https://news.ycombinator.com/",
        page_key="hn:front",
        action_json='{"action":"done"}',
        visible_tokens=212,
        baseline_tokens=14200,
        bank_found=True,
    )
    res = client.query(
        f"SELECT count() FROM {storage.STEPS_TABLE} "
        f"WHERE session_id = 'sess-test'"
    )
    assert res.result_rows[0][0] >= 1
    client.command(
        f"ALTER TABLE {storage.STEPS_TABLE} DELETE WHERE session_id = 'sess-test'"
    )
