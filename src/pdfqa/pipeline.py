"""The RAG pipeline: index PDFs, then answer questions with citations.

Flow for each question:
  1. Condense the follow-up into a standalone question (uses chat history)
  2. Hybrid-retrieve the top-k chunks
  3. Ask the LLM to answer from those chunks only, citing [n]
If no LLM is configured, it falls back to showing the best passages
(extractive mode), so the app still works without an API key.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .config import Settings
from .ingest import PDFFile, ingest
from .prompts import ANSWER_PROMPT, CONDENSE_PROMPT
from .retriever import HybridRetriever
from .vector_store import VectorStore


@dataclass
class Answer:
    question: str            # standalone question actually used for retrieval
    sources: list[Document]
    stream: Iterator[str]    # answer text, streamed token by token


@dataclass
class Turn:
    question: str
    answer: str


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


@dataclass
class RAGPipeline:
    settings: Settings
    embeddings: Embeddings
    llm: BaseChatModel | None = None
    retriever: HybridRetriever | None = None
    history: list[Turn] = field(default_factory=list)

    # ---------- indexing ----------
    def _cache_key(self, pdfs: list[PDFFile]) -> str:
        s = self.settings
        parts = sorted(p.sha256 for p in pdfs) + [s.embedding_model, str(s.chunk_size), str(s.chunk_overlap)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def index(self, pdfs: list[PDFFile], use_cache: bool = True) -> int:
        """Build (or load from cache) the index for these PDFs. Returns the chunk count."""
        folder = self.settings.cache_dir / self._cache_key(pdfs)
        if use_cache and VectorStore.exists(folder):
            store = VectorStore.load(folder, self.embeddings)
        else:
            chunks = ingest(pdfs, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                raise ValueError("No text found in the PDFs. Scanned (image-only) PDFs need OCR.")
            store = VectorStore.build(chunks, self.embeddings)
            if use_cache:
                store.save(folder)
        self.retriever = HybridRetriever(store, self.settings.candidate_k)
        self.history.clear()
        return len(store.chunks)

    # ---------- question answering ----------
    def _condense(self, question: str) -> str:
        if not self.history or self.llm is None:
            return question
        recent = self.history[-self.settings.history_turns:]
        history = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in recent)
        chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"history": history, "question": question}).strip() or question

    @staticmethod
    def _extractive(sources: list[Document]) -> Iterator[str]:
        yield "_No API key set, so here are the most relevant passages:_\n\n"
        for i, d in enumerate(sources[:3], start=1):
            yield f"**[{i}]** {d.page_content[:400]}…\n\n"

    def ask(self, question: str) -> Answer:
        if self.retriever is None:
            raise RuntimeError("Call index() before ask().")
        standalone = self._condense(question)
        sources = self.retriever.retrieve(standalone, k=self.settings.top_k)
        if self.llm is None:
            stream = self._extractive(sources)
        else:
            chain = ANSWER_PROMPT | self.llm | StrOutputParser()
            stream = chain.stream({"context": format_sources(sources), "question": standalone})
        return Answer(question=standalone, sources=sources, stream=stream)

    def remember(self, question: str, answer: str) -> None:
        self.history.append(Turn(question, answer))
