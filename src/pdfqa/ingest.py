"""PDF loading and chunking.

Each chunk is a LangChain ``Document`` whose metadata keeps the source file,
the page number and a stable ``chunk_id`` so answers can cite their sources.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class PDFFile:
    name: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def load_pdf(pdf: PDFFile) -> list[Document]:
    """Return one Document per non-empty page."""
    reader = PdfReader(io.BytesIO(pdf.data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append(Document(page_content=text, metadata={"source": pdf.name, "page": i}))
    return pages


def chunk_documents(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into overlapping chunks, preferring paragraph/sentence boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest(pdfs: list[PDFFile], chunk_size: int, chunk_overlap: int) -> list[Document]:
    pages = [page for pdf in pdfs for page in load_pdf(pdf)]
    return chunk_documents(pages, chunk_size, chunk_overlap)
