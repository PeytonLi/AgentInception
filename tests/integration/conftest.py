"""Integration test infrastructure. C1 brief §Method.

Adds all four app source roots + shared-py + mocks to sys.path so every test
imports directly from the source trees — no editable installs required.

Slow / GPU tests are guarded by pytest markers; CI (and off-GPU dev machines)
skips them automatically unless --run-slow / --run-gpu is passed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent  # tests/integration
_REPO = _HERE.parent.parent  # worktree root

for _p in (
    _REPO / "packages" / "shared-py",
    _REPO / "apps" / "inference-engine" / "src",
    _REPO / "apps" / "agent-runner",
    _REPO / "apps" / "bank-compiler" / "src",
    _REPO / "tests" / "mocks",
):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False)
    parser.addoption("--run-gpu", action="store_true", default=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring a CUDA GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow to execute")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu to execute (requires CUDA)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO


@pytest.fixture(scope="session")
def banks_dir(repo_root: Path) -> Path:
    d = repo_root / "banks"
    if not (d / "manifest.json").exists():
        pytest.skip("banks/manifest.json not found — run scripts/build_demo_banks.py first")
    return d


@pytest.fixture(scope="session")
def manifest(banks_dir: Path) -> dict:
    from ghost_shared import bank_io
    return bank_io.read_manifest(str(banks_dir))


REQUIRED_PAGE_KEYS = {"hn:front", "hn:item", "popup:demo"}
