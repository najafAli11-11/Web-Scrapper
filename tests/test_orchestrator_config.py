"""Config tests for the orchestrator retry policy (M7 commit 1)."""

import json

import pytest

from orchestrator.config_loader import load_orchestrator_config


def test_orchestrator_config_validates():
    cfg = load_orchestrator_config()
    assert cfg["retry"]["max_attempts"] >= 1
    assert cfg["retry"]["backoff_seconds"] >= 0


def test_orchestrator_config_rejects_missing_retry(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"something_else": 1}))
    with pytest.raises(ValueError):
        load_orchestrator_config(bad)


def test_orchestrator_config_rejects_zero_max_attempts(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"retry": {"max_attempts": 0, "backoff_seconds": 5}}))
    with pytest.raises(ValueError):
        load_orchestrator_config(bad)


def test_orchestrator_config_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"retry": {"max_attempts": 2, "backoff_seconds": 5}, "junk": 1}))
    with pytest.raises(ValueError):
        load_orchestrator_config(bad)
