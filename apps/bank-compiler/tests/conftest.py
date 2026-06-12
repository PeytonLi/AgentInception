"""Make bank_compiler and ghost_shared importable without editable installs."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_APP = _HERE.parents[1]
_REPO = _HERE.parents[3]

sys.path.insert(0, str(_APP / "src"))
sys.path.insert(0, str(_REPO / "packages" / "shared-py"))
