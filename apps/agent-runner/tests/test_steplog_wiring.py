"""A3 <-> shared-py wiring: StepLogger persists agent_steps via ghost_shared.

These tests exercise the cross-component seam (CONTRACTS s5) through the public
StepLogger interface, so they catch any drift in ghost_shared.storage.log_step
(table name, column order, value coercion) - the kind of breakage a branch
merge can introduce silently.
"""

from __future__ import annotations

from agent_runner.steplog import StepLogger


class _FakeClickHouse:
    """Records inserts the way clickhouse-connect's client would receive them."""

    def __init__(self) -> None:
        self.inserts: list[tuple] = []

    def command(self, sql: str) -> int:  # used by StepLogger.connect health-check
        return 1

    def insert(self, table, rows, column_names):  # noqa: ANN001
        self.inserts.append((table, rows, column_names))


def _log_one(logger: StepLogger, **overrides):
    row = {
        "session_id": "s1",
        "step": 0,
        "mode": "mi",
        "url": "https://news.ycombinator.com/",
        "page_key": "hn:front",
        "action_json": '{"action": "goto", "url": "x"}',
        "visible_tokens": 210,
        "baseline_tokens": 14200,
        "bank_found": True,
    }
    row.update(overrides)
    logger.log(**row)
    return row


def test_steplog_forwards_to_clickhouse_with_contract_columns():
    client = _FakeClickHouse()
    logger = StepLogger(client)

    _log_one(logger)

    assert len(logger.rows) == 1  # always kept in-memory too
    assert len(client.inserts) == 1
    table, rows, columns = client.inserts[0]
    assert table == "ghostbrowser.agent_steps"
    assert columns == [
        "session_id", "step", "mode", "url", "page_key",
        "action_json", "visible_tokens", "baseline_tokens", "bank_found",
    ]
    row = rows[0]
    assert row[0] == "s1"
    assert row[2] == "mi"
    assert row[4] == "hn:front"
    assert row[8] == 1  # bank_found True -> UInt8 1 (CONTRACTS s5)


def test_steplog_degrades_to_in_memory_when_clickhouse_unavailable():
    # No reachable ClickHouse / no clickhouse-connect installed -> no-op client.
    logger = StepLogger.connect(url="http://127.0.0.1:1", enabled=True)
    _log_one(logger)
    assert logger._client is None
    assert len(logger.rows) == 1  # still recorded, no crash


def test_steplog_disabled_never_connects():
    logger = StepLogger.connect(enabled=False)
    _log_one(logger)
    assert logger._client is None
    assert len(logger.rows) == 1
