"""Hybrid retrieval: BM25 (keywords) + FAISS (meaning), fused with Reciprocal Rank Fusion.

Why hybrid? Vector search understands paraphrases ("cost" ~ "price") but can
miss exact terms such as error codes, names or numbers. BM25 catches exact
terms but not meaning. RRF combines the two ranked lists without having to
calibrate their very different score scales.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids. score(id) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, candidate_k: int = 20):
        self.store = store
        self.candidate_k = candidate_k
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in store.chunks])

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[: self.candidate_k] if scores[i] > 0]

    def _vector_ranking(self, query: str) -> list[int]:
        return [doc.metadata["chunk_id"] for doc, _ in self.store.search(query, self.candidate_k)]

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        fused = reciprocal_rank_fusion([self._vector_ranking(query), self._bm25_ranking(query)])
        by_id = {c.metadata["chunk_id"]: c for c in self.store.chunks}
        results = []
        for doc_id, score in fused[:k]:
            doc = by_id[doc_id]
            results.append(Document(page_content=doc.page_content,
                                    metadata={**doc.metadata, "score": round(score, 4)}))
        return results
