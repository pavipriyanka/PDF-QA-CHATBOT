from pdfqa.ingest import ingest
from pdfqa.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize
from pdfqa.vector_store import VectorStore


def test_tokenize():
    assert tokenize("Ticket IT-HW-07, please!") == ["ticket", "it", "hw", "07", "please"]


def test_rrf_rewards_agreement():
    # doc 2 is ranked well by both lists, so it should win
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]])
    assert fused[0][0] == 2
    assert {doc for doc, _ in fused} == {1, 2, 3, 4}


def test_bm25_finds_exact_code(sample_pdf, fake_embeddings):
    chunks = ingest([sample_pdf], chunk_size=300, chunk_overlap=50)
    retriever = HybridRetriever(VectorStore.build(chunks, fake_embeddings))
    top = retriever.retrieve("ticket code IT-HW-07", k=3)
    assert any("IT-HW-07" in d.page_content for d in top)
    assert all("score" in d.metadata for d in top)


def test_vector_store_roundtrip(tmp_path, sample_pdf, fake_embeddings):
    chunks = ingest([sample_pdf], chunk_size=300, chunk_overlap=50)
    store = VectorStore.build(chunks, fake_embeddings)
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx", fake_embeddings)
    q = "parental leave"
    assert [d.page_content for d, _ in store.search(q, 3)] == [d.page_content for d, _ in loaded.search(q, 3)]
