"""Medical corpus loading and chunking for the demo's RAG retriever.

Pipeline 2, demo only. None of the experimental results touch this module.

Chunking is fixed-size with overlap - deliberately unsophisticated. Section
"explicitly out of scope" is clear that retrieval quality is not evaluated: no
recall@k, no chunking ablation. The generation pipeline is infrastructure, not
an object of study, so the chunker only has to be good enough that grounded
answers beat a bare model hallucinating in front of an examiner.

Embedding happens through an injected function, same as ``cache.py``, so this
module stays free of model imports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "medical"


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, carrying enough to cite it in the demo."""

    chunk_id: int
    text: str
    source: str
    title: str = ""

    def citation(self) -> str:
        return f"{self.title} ({self.source})" if self.title else self.source


def chunk_text(
    text: str,
    source: str,
    title: str = "",
    target_words: int = 120,
    overlap_words: int = 20,
    start_id: int = 0,
) -> list[Chunk]:
    """Split ``text`` into overlapping word windows.

    Overlap exists so a passage split mid-explanation is still retrievable from
    either side of the boundary.
    """
    if target_words <= overlap_words:
        raise ValueError("target_words must exceed overlap_words")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = target_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + target_words]
        if len(window) < 20 and chunks:
            break  # trailing scrap; the previous chunk's overlap already covers it
        chunks.append(
            Chunk(
                chunk_id=start_id + len(chunks),
                text=" ".join(window),
                source=source,
                title=title,
            )
        )
    return chunks


def load_corpus(
    corpus_dir: Path | str | None = None,
    target_words: int = 120,
    overlap_words: int = 20,
) -> list[Chunk]:
    """Load and chunk every document under ``corpus_dir``.

    Accepts two shapes so the demo is not blocked on one dataset's format:

      *.txt   - one document per file; the filename becomes the title
      *.json  - a list of {"title", "text", "source"} objects, which is the
                shape MedQuAD and MedlinePlus exports are easiest to flatten to

    Whichever medical source is used, converting it to one of these two shapes
    is a preprocessing step, kept out of here so this module has no dataset
    schema baked into it.
    """
    directory = Path(corpus_dir) if corpus_dir is not None else _DEFAULT_CORPUS_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"medical corpus directory not found at {directory}. Populate it with "
            f".txt or .json documents, or pass corpus_dir explicitly."
        )

    chunks: list[Chunk] = []

    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for document in payload:
            chunks.extend(
                chunk_text(
                    text=document.get("text", ""),
                    source=document.get("source", path.name),
                    title=document.get("title", ""),
                    target_words=target_words,
                    overlap_words=overlap_words,
                    start_id=len(chunks),
                )
            )

    for path in sorted(directory.glob("*.txt")):
        chunks.extend(
            chunk_text(
                text=path.read_text(encoding="utf-8"),
                source=path.name,
                title=_title_from_filename(path.stem),
                target_words=target_words,
                overlap_words=overlap_words,
                start_id=len(chunks),
            )
        )

    if not chunks:
        raise ValueError(f"no .txt or .json documents found in {directory}")
    return chunks


def _title_from_filename(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip().title()


def embed_chunks(
    chunks: list[Chunk],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Embed every chunk in one batched call; returns ``(len(chunks), dim)``."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(embed_fn([c.text for c in chunks]), dtype=np.float32)
