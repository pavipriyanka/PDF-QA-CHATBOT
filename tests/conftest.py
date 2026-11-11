import sys
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pdfqa import PDFFile, settings  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> PDFFile:
    import make_sample_pdf

    path = make_sample_pdf.main(tmp_path_factory.mktemp("data") / "sample.pdf")
    return PDFFile("sample.pdf", path.read_bytes())


@pytest.fixture
def fake_embeddings():
    # Offline, deterministic stand-in for the real HuggingFace model.
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def test_settings(tmp_path):
    return replace(settings, cache_dir=tmp_path / "cache", chunk_size=300, chunk_overlap=50, top_k=3)
