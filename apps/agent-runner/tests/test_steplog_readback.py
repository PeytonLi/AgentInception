"""P3 task 5: agent_steps rows can be queried back from ClickHouse.

A fake ClickHouse captures the insert and serves it back through ``.query()``,
exercising StepLogger.read_back's column mapping without a live DB. A second,
``@pytest.mark.clickhouse`` test runs the real round-trip when CLICKHOUSE_URL
points at a reachable server (skipped otherwise, so CI stays green).
"""

from __future__ import annotations

import os

import pytest

from agent_runner.steplog import StepLogger


class _QueryResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeClickHouse:
    def __init__(self):
        self.rows = []

    def command(self, sql):
        return 1

    def insert(self, table, rows, column_names):
        self.rows.extend(rows)

    def query(self, sql, parameters=None):
        sid = parameters["sid"]
        return _QueryResult([r for r in self.rows if r[0] == sid])


def _log(logger, step, **kw):
    base = dict(
        session_id="s-read",
        step=step,
        mode="mi",
        url="https://news.ycombinator.com/",
        page_key="hn:front",
        action_json='{"action": "goto", "url": "x"}',
        visible_tokens=210,
        baseline_tokens=14200,
        bank_found=True,
    )
    base.update(kw)
    logger.log(**base)


def test_read_back_maps_columns_from_clickhouse():
    client = _FakeClickHouse()
    logger = StepLogger(client)
    _log(logger, 0)
    _log(logger, 1, page_key="hn:item", action_json='{"action": "extract", "result": {}}')

    rows = logger.read_back("s-read")
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["mode"] == "mi"
    assert rows[0]["page_key"] == "hn:front"
    assert rows[1]["page_key"] == "hn:item"
    assert rows[0]["bank_found"] in (1, True)


def test_read_back_falls_back_to_memory_without_client():
    logger = StepLogger(None)
    _log(logger, 0)
    rows = logger.read_back("s-read")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s-read"


@pytest.mark.clickhouse
def test_real_clickhouse_round_trip():
    url = os.environ.get("CLICKHOUSE_URL")
    if not url:
        pytest.skip("CLICKHOUSE_URL not set")
    logger = StepLogger.connect(url=url, enabled=True)
    if logger._client is None:
        pytest.skip("ClickHouse unreachable")
    import uuid

    sid = f"p3-readback-{uuid.uuid4().hex[:8]}"
    _log(logger, 0, session_id=sid)
    _log(logger, 1, session_id=sid, page_key="hn:item")
    rows = logger.read_back(sid)
    assert [r["step"] for r in rows] == [0, 1]
