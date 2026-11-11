"""A small FAISS vector store with save/load to disk.

Embeddings are L2-normalised, so inner product == cosine similarity.
Chunks are stored as JSON next to the index (no pickle, safe to load).
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _normalise(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Document], embeddings: Embeddings):
        self.index = index
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def build(cls, chunks: list[Document], embeddings: Embeddings) -> "VectorStore":
        vectors = _normalise(embeddings.embed_documents([c.page_content for c in chunks]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embeddings)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        q = _normalise([self.embeddings.embed_query(query)])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        data = [{"text": c.page_content, "metadata": c.metadata} for c in self.chunks]
        (folder / "chunks.json").write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embeddings: Embeddings) -> "VectorStore":
        index = faiss.read_index(str(folder / "index.faiss"))
        data = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Document(page_content=d["text"], metadata=d["metadata"]) for d in data]
        return cls(index, chunks, embeddings)

    @staticmethod
    def exists(folder: Path) -> bool:
        return (folder / "index.faiss").exists() and (folder / "chunks.json").exists()
