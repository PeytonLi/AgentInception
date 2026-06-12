"""ASGI entrypoint for `uvicorn src.main:app` (see docs/handoff/peyton/aws-runbook.md section 6).

create_app() is cheap: the real Llama backend and bank registry load lazily in
the FastAPI lifespan on first startup, so importing this module never touches the
GPU. Equivalent to `python -m inference_engine.server`, but lets uvicorn own the
process (workers / reload / signal handling).
"""

from inference_engine.config import Settings
from inference_engine.server import create_app

app = create_app(settings=Settings.from_env())
