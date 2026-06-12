"""B2 spec, unit test #2: running scripts/upload_banks.py twice against a
local ClickHouse must leave the row count unchanged (delete-then-insert by
page_key).

Skipped automatically when ClickHouse is not reachable — run
`scripts/ch_init.sh` first, then re-run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentinception_shared import bank_io, storage

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.fixture(scope="module")
def banks_dir(tmp_path_factory) -> Path:
    repo_dir = REPO_ROOT / "banks"
    if repo_dir.exists() and any(repo_dir.glob("*.bin")):
        return repo_dir
    tmp_dir = tmp_path_factory.mktemp("banks")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_demo_banks.py"),
            "--force-synthetic",
            "--out",
            str(tmp_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return tmp_dir


@pytest.fixture
def cleaned_clickhouse():
    """Delete the demo page_keys from ClickHouse before AND after the test."""
    client = _client_or_skip()
    demo_keys = ("hn:front", "hn:item", "popup:demo")
    placeholders = ",".join(f"'{k}'" for k in demo_keys)
    sql = (
        f"ALTER TABLE {storage.BANKS_TABLE} DELETE "
        f"WHERE page_key IN ({placeholders})"
    )
    client.command(sql)
    yield client
    client.command(sql)


def _count_rows_for(client, page_keys: tuple[str, ...]) -> int:
    placeholders = ",".join(f"'{k}'" for k in page_keys)
    res = client.query(
        f"SELECT count() FROM {storage.BANKS_TABLE} "
        f"WHERE page_key IN ({placeholders})"
    )
    return int(res.result_rows[0][0])


@pytest.mark.integration
def test_upload_idempotent(banks_dir, cleaned_clickhouse):
    client = cleaned_clickhouse
    upload = REPO_ROOT / "scripts" / "upload_banks.py"
    demo_keys = ("hn:front", "hn:item", "popup:demo")

    manifest = bank_io.read_manifest(str(banks_dir))
    expected_rows = sum(len(e["files"]) for e in manifest["banks"])  # 3 banks × 4 layers = 12

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "packages" / "shared-py")}

    first = subprocess.run(
        [sys.executable, str(upload), str(banks_dir)],
        check=True, capture_output=True, text=True, env=env,
    )
    assert "uploaded" in first.stdout, first.stdout

    rows_after_first = _count_rows_for(client, demo_keys)
    assert rows_after_first == expected_rows, (
        f"after first upload expected {expected_rows} rows, "
        f"got {rows_after_first}"
    )

    subprocess.run(
        [sys.executable, str(upload), str(banks_dir)],
        check=True, capture_output=True, text=True, env=env,
    )
    rows_after_second = _count_rows_for(client, demo_keys)

    assert rows_after_second == rows_after_first, (
        f"upload not idempotent: {rows_after_first} -> {rows_after_second}"
    )
