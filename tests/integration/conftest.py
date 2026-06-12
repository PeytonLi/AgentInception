"""Integration test infrastructure. C1 brief §Method.

Adds all four app source roots + shared-py + mocks to sys.path so every test
imports directly from the source trees — no editable installs required.

Slow / GPU tests are guarded by pytest markers; CI (and off-GPU dev machines)
skips them automatically unless --run-slow / --run-gpu is passed.
"""

from __future__ import annotations

import importlib.util
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


_APP_FAKES = {
    "inference-engine": _REPO / "apps" / "inference-engine" / "tests" / "fakes.py",
    "agent-runner": _REPO / "apps" / "agent-runner" / "tests" / "fakes.py",
}


def import_app_fakes(app: str):
    """Load an app's ``tests/fakes.py`` under a unique, namespaced module name.

    Both the inference-engine and the agent-runner ship a test-double module
    literally named ``fakes``. Importing them with the bare name ``fakes`` makes
    whichever one loads second collide with the first in ``sys.modules`` — the
    bug that stranded integration tests 7-9 & 11 during collection. Loading each
    under ``_appfakes_<app>`` keeps the two independent regardless of collection
    order.
    """
    path = _APP_FAKES[app]
    mod_name = f"_appfakes_{app.replace('-', '_')}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load fakes for {app} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False)
    parser.addoption("--run-gpu", action="store_true", default=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring a CUDA GPU")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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
        pytest.skip(
            "banks/manifest.json not found — run scripts/build_demo_banks.py first"
        )
    return d


@pytest.fixture(scope="session")
def manifest(banks_dir: Path) -> dict:
    from ghost_shared import bank_io

    return bank_io.read_manifest(str(banks_dir))


REQUIRED_PAGE_KEYS = {"hn:front", "hn:item", "popup:demo"}
