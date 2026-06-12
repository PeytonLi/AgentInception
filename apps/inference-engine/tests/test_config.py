"""Constants must match CONTRACTS.md §1 exactly — drift here breaks B1's banks."""

from inference_engine import config


def test_contract_constants():
    assert config.MODEL_ID == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.SELECTED_LAYERS == [8, 12, 16, 20]
    assert config.NUM_LAYERS == 32
    assert config.NUM_Q_HEADS == 32
    assert config.NUM_KV_HEADS == 8
    assert config.HEAD_DIM == 128
    assert config.HIDDEN_SIZE == 4096


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_URL", "http://somewhere:8123")
    monkeypatch.setenv("BANKS_DIR", "X:/banks")
    monkeypatch.setenv("MODEL_ID", "tiny/test-model")
    s = config.Settings.from_env()
    assert s.clickhouse_url == "http://somewhere:8123"
    assert s.banks_dir == "X:/banks"
    assert s.model_id == "tiny/test-model"


def test_settings_defaults(monkeypatch):
    for var in ("CLICKHOUSE_URL", "BANKS_DIR", "MODEL_ID", "INFERENCE_PORT"):
        monkeypatch.delenv(var, raising=False)
    s = config.Settings.from_env()
    assert s.clickhouse_url == "http://localhost:8123"
    assert s.model_id == config.MODEL_ID
    assert s.port == 8000
