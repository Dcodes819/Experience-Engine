"""Swappable embedding providers with a deterministic offline default."""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Protocol


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbeddingProvider:
    """Stable bag-of-words embeddings for a reproducible, offline MVP baseline."""

    name = "hashing-v1-256d"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9_/-]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class SentenceTransformerEmbeddingProvider:
    """Optional local semantic embeddings, loaded only when explicitly configured."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Sentence Transformers is optional; install the 'semantic' extra to use this provider."
            ) from error
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache(maxsize=1)
def configured_embedding_provider() -> EmbeddingProvider:
    """Return one process-wide configured provider, defaulting to offline hashing."""
    provider = os.getenv("EXPERIENCE_ENGINE_EMBEDDING_PROVIDER", "hashing").strip().lower()
    if provider == "hashing":
        return HashingEmbeddingProvider()
    if provider in {"sentence-transformers", "sentence_transformers"}:
        model_name = os.getenv("EXPERIENCE_ENGINE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingProvider(model_name)
    raise ValueError(f"Unsupported EXPERIENCE_ENGINE_EMBEDDING_PROVIDER: {provider}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))
