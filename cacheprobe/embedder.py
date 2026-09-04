"""Sentence embedding with an on-disk cache.

Embedding is the slowest step in the pipeline and is identical across every
eviction policy under comparison, so each vector is computed once and reused
from disk thereafter.

Vectors are L2-normalised by default, which makes cosine similarity a plain dot
product. Both the LSH index (random hyperplanes separate by angle) and the
cache's similarity threshold assume this.

``sentence_transformers`` is imported lazily, inside ``_encode``. Importing
torch is expensive, and nothing else in the project needs it - ``traces.py``,
``cache.py`` and the whole test suite work on plain arrays, so they must be
able to import this module without paying for a transformer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


def _key(text: str) -> str:
    """Stable content hash used as the disk-cache key for one string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Embedder:
    """Encodes text to dense vectors, memoised on disk.

    The cache is two files: a JSON list of content hashes and a 2D ``.npy``
    array whose row *i* holds the vector for hash *i*.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: Path | str | None = None,
        normalise: bool = True,
        batch_size: int = 256,
        autosave: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalise = normalise
        self.batch_size = batch_size
        self.autosave = autosave

        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        stem = model_name.replace("/", "_") + (".norm" if normalise else "")
        self._keys_path = self.cache_dir / f"{stem}.keys.json"
        self._vecs_path = self.cache_dir / f"{stem}.vecs.npy"

        self._model = None  # lazily constructed; importing torch is expensive
        self._index: dict[str, int] | None = None
        self._vecs: np.ndarray | None = None
        self._dirty = False

    # ------------------------------------------------------------------ cache

    def _load(self) -> None:
        """Read the disk cache into memory on first use."""
        if self._index is not None:
            return

        if self._keys_path.exists() and self._vecs_path.exists():
            keys = json.loads(self._keys_path.read_text(encoding="utf-8"))
            vecs = np.load(self._vecs_path)
            if len(keys) != len(vecs):
                raise ValueError(
                    f"embedding cache is inconsistent: {len(keys)} keys vs "
                    f"{len(vecs)} vectors in {self.cache_dir}"
                )
            self._index = {k: i for i, k in enumerate(keys)}
            self._vecs = vecs
        else:
            self._index = {}
            self._vecs = np.empty((0, EMBED_DIM), dtype=np.float32)

    def save(self) -> None:
        """Persist any newly computed vectors."""
        if not self._dirty or self._index is None or self._vecs is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys: list[str | None] = [None] * len(self._index)
        for k, i in self._index.items():
            keys[i] = k
        self._keys_path.write_text(json.dumps(keys), encoding="utf-8")
        np.save(self._vecs_path, self._vecs)
        self._dirty = False

    # ------------------------------------------------------------------ model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalise,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ----------------------------------------------------------------- public

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), EMBED_DIM)`` array, one row per input.

        Rows come back in input order. Repeated strings cost nothing beyond the
        first occurrence, within a call and across runs.
        """
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        self._load()
        assert self._index is not None and self._vecs is not None

        missing: list[str] = []
        pending: set[str] = set()
        for text in texts:
            k = _key(text)
            if k not in self._index and k not in pending:
                pending.add(k)
                missing.append(text)

        if missing:
            new_vecs = self._encode(missing)
            start = len(self._vecs)
            self._vecs = np.vstack([self._vecs, new_vecs])
            for offset, text in enumerate(missing):
                self._index[_key(text)] = start + offset
            self._dirty = True
            if self.autosave:
                self.save()

        rows = [self._index[_key(text)] for text in texts]
        return self._vecs[rows]

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single ``(EMBED_DIM,)`` vector."""
        return self.embed([text])[0]


_default: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide default embedder, so the disk cache is shared."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().embed(texts)


def embed_one(text: str) -> np.ndarray:
    return get_embedder().embed_one(text)
