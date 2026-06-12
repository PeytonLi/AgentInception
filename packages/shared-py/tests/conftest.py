import sys
from pathlib import Path

# Ensure the package is importable when running pytest from the repo root
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
