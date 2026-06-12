import sys

sys.path.insert(0, "apps/inference-engine/tests")
from fakes import FakeBackend
from fastapi.testclient import TestClient

from inference_engine.bank_registry import BankRegistry
from inference_engine.server import create_app

reg = BankRegistry.load("http://localhost:9", "tests/fixtures/tiny_bank/generated")
print("source:", reg.source, "| page_keys:", reg.page_keys, "| slots:", reg.num_slots("hn:front"))
client = TestClient(create_app(backend=FakeBackend(), registry=reg))
print("healthz:", client.get("/healthz").json())
