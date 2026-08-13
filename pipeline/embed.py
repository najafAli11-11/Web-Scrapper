"""Local embedding (Spec req. 12) — BGE-M3 default, all-MiniLM-L6-v2 fallback.

Fallback trigger is explicit and config-driven (AGENTS.md tech conventions,
no silent hardware sniffing): load_embedder tries to load `embed.model` from
config/embeddings.json; only on an actual load failure does it (if
`auto_fallback_on_load_failure`) log an embed_model_fallback event and load
`fallback_model`. Manual override = set `embed.model` directly.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Protocol

from fetchers.logger import FetchLogger


class Embedder(Protocol):
    """Embedding backend contract (mirrors agents/llm/client.py's protocol pattern)."""

    model_name: str
    dimension: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a dense normalized vector per input text."""
        ...


_DIM_CACHE: dict[str, int] = {}


class LocalEmbedder:
    """sentence-transformers-backed embedder (dense, normalized)."""

    def __init__(self, model_name: str, *, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dimension = _probe_dimension(self._model, model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(v) for v in vecs]


def _probe_dimension(model, model_name: str) -> int:
    if model_name in _DIM_CACHE:
        return _DIM_CACHE[model_name]
    vec = model.encode(["probe"], normalize_embeddings=True)
    dim = len(vec[0])
    _DIM_CACHE[model_name] = dim
    return dim


def load_embedder(
    cfg: dict,
    *,
    logger: Optional[FetchLogger] = None,
    model_factory: Optional[Callable[[str], Embedder]] = None,
) -> Embedder:
    """Build the configured embedder, applying the explicit config-gated fallback.

    `model_factory` is injectable for hermetic tests (default: LocalEmbedder).
    """
    embed_cfg = cfg.get("embed", {})
    model = embed_cfg.get("model", "BAAI/bge-m3")
    fallback = embed_cfg.get("fallback_model", "all-MiniLM-L6-v2")
    auto_fallback = bool(embed_cfg.get("auto_fallback_on_load_failure", True))
    factory = model_factory or (lambda name: LocalEmbedder(name, batch_size=embed_cfg.get("batch_size", 32)))

    try:
        return factory(model)
    except Exception as exc:  # noqa: BLE001 - load failure is the fallback trigger
        if not auto_fallback:
            raise
        if logger is not None:
            logger.log_event(
                event_type="embed_model_fallback",
                url=None,
                outcome="fallback",
                reason=f"{model} failed to load, using {fallback}",
                details={"model": model, "fallback_model": fallback, "error": str(exc)[:500]},
            )
        else:
            print(
                f"[embed] model '{model}' failed to load ({exc}); "
                f"falling back to '{fallback}'",
                file=sys.stderr,
            )
        try:
            return factory(fallback)
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                f"embedding models unavailable: primary '{model}' failed ({exc}) "
                f"and fallback '{fallback}' failed ({exc2})"
            ) from exc2
