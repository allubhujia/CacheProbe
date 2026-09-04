"""RAG retrieval over document chunks, reusing the project's own LSH index.

No external vector library. The same structure that finds a cached query's
nearest neighbour finds a question's relevant passages - which is the point
worth making in the report: one hash-based index, two jobs.

Retrieval quality is explicitly not evaluated. This exists so the demo's
answers are grounded in a real document instead of invented, and so the miss
path is long enough that the hit/miss gap is visually obvious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..corpus import Chunk
from ..lsh_index import LSHIndex


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


class ChunkRetriever:
    """Top-k passage retrieval backed by ``LSHIndex``.

    LSH narrows to a candidate set, then candidates are ranked exactly by
    cosine similarity - the same two-stage shape as the cache's lookup.
    ``exhaustive_fallback`` re-scans everything when the probe returns too few
    candidates, which matters more here than in the cache: an empty candidate
    set means an ungrounded answer, whereas in the cache it merely means a miss.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        k: int = 12,
        L: int = 8,
        seed: int = 0,
        exhaustive_fallback: bool = True,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.exhaustive_fallback = exhaustive_fallback

        dim = self.vectors.shape[1] if len(self.vectors) else 0
        self.index: LSHIndex[int] = LSHIndex(dim=dim, k=k, L=L, seed=seed)
        for i in range(len(chunks)):
            self.index.insert(i, self.vectors[i])

    def __len__(self) -> int:
        return len(self.chunks)

    def retrieve(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievedChunk]:
        # An empty corpus is a real state, not a misuse: the medical documents
        # are a separate download, so the demo can be started before any exist.
        # Returning nothing lets the caller answer "no reference material"
        # rather than crashing on a zero-width index.
        if not self.chunks:
            return []

        v = np.asarray(query_vector, dtype=np.float32)
        candidates = self.index.probe(v)

        if self.exhaustive_fallback and len(candidates) < top_k:
            candidates = set(range(len(self.chunks)))

        scored = [(i, float(np.dot(v, self.vectors[i]))) for i in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedChunk(chunk=self.chunks[i], similarity=sim) for i, sim in scored[:top_k]
        ]

    def build_context(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        max_chars: int = 4000,
    ) -> tuple[str, tuple[str, ...]]:
        """Assemble retrieved passages into a prompt context plus citations.

        Returns the context block and the source list the demo displays, so an
        answer is traceable to a document rather than taken on trust.
        """
        retrieved = self.retrieve(query_vector, top_k=top_k)

        parts: list[str] = []
        sources: list[str] = []
        used = 0
        for item in retrieved:
            block = f"[{item.chunk.citation()}]\n{item.chunk.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            if item.chunk.citation() not in sources:
                sources.append(item.chunk.citation())
            used += len(block)

        return "\n\n".join(parts), tuple(sources)


def build_retriever(
    chunks: list[Chunk],
    embed_fn: Callable[[list[str]], np.ndarray],
    **kwargs: object,
) -> ChunkRetriever:
    """Embed a chunk list and wrap it in a retriever."""
    vectors = np.asarray(embed_fn([c.text for c in chunks]), dtype=np.float32)
    return ChunkRetriever(chunks, vectors, **kwargs)  # type: ignore[arg-type]
