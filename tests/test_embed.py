"""Hermetic tests for Milestone 6 embedding (pipeline/embed.py).

No real model and no network: a fake Embedder factory is injected so model
selection, the config-gated fallback, the embed_model_fallback event, and
dimension probing are tested without loading weights.
"""

from __future__ import annotations

import pytest

from fetchers.logger import FetchLogger
from pipeline.config_loader import load_pipeline_config
from pipeline.embed import load_embedder


class FakeEmbedder:
    def __init__(self, name: str, dim: int = 8):
        self.model_name = name
        self.dimension = dim
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[0.1] * self.dimension for _ in texts]


class FailingEmbedder:
    def __init__(self, name: str):
        raise RuntimeError(f"cannot load {name}")


def _cfg(model="BAAI/bge-m3", fallback="all-MiniLM-L6-v2", auto=True, **extra):
    return {
        "embed": {
            "model": model,
            "fallback_model": fallback,
            "auto_fallback_on_load_failure": auto,
            "batch_size": 32,
            **extra,
        }
    }


def test_default_config_selects_bge_m3():
    cfg = load_pipeline_config()
    assert cfg["embed"]["model"] == "BAAI/bge-m3"
    assert cfg["embed"]["fallback_model"] == "all-MiniLM-L6-v2"


def test_load_embedder_builds_configured_model():
    built: list[str] = []
    def factory(name):
        built.append(name)
        return FakeEmbedder(name)
    emb = load_embedder(_cfg(model="BAAI/bge-m3"), model_factory=factory)
    assert built == ["BAAI/bge-m3"]
    assert emb.model_name == "BAAI/bge-m3"
    assert emb.dimension == 8
    assert emb.embed(["a", "b"]) == [[0.1] * 8, [0.1] * 8]


def test_fallback_triggers_on_load_failure_when_enabled(tmp_path):
    built: list[str] = []
    def factory(name):
        built.append(name)
        if name == "BAAI/bge-m3":
            raise RuntimeError("boom")
        return FakeEmbedder(name)
    with FetchLogger(tmp_path / "logs.db") as logger:
        emb = load_embedder(_cfg(), logger=logger, model_factory=factory)
        rows = logger.recent_events()
    assert built == ["BAAI/bge-m3", "all-MiniLM-L6-v2"]
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert any(r["event_type"] == "embed_model_fallback" for r in rows)


def test_fallback_disabled_reraises(tmp_path):
    def factory(name):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        load_embedder(_cfg(auto=False), model_factory=factory)


def test_fallback_failure_raises_runtime_error(tmp_path):
    def factory(name):
        raise RuntimeError(f"never {name}")
    with pytest.raises(RuntimeError, match="unavailable"):
        load_embedder(_cfg(), model_factory=factory)


def test_empty_texts_embed_to_empty():
    emb = FakeEmbedder("x")
    assert emb.embed([]) == []


def test_local_embedder_converts_numpy_floats_to_python_floats(monkeypatch):
    import numpy as np

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[np.float32(0.1), np.float32(0.2)], [np.float32(0.3), np.float32(0.4)]])

    import pipeline.embed as embed_mod
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda name: FakeModel())
    emb = embed_mod.LocalEmbedder("fake-model")
    result = emb.embed(["a", "b"])
    assert np.allclose(result, [[0.1, 0.2], [0.3, 0.4]], atol=1e-6)
    assert all(type(x) is float for vec in result for x in vec)
