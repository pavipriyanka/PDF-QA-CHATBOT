from pdfqa.ingest import chunk_documents, ingest, load_pdf


def test_load_pdf_keeps_page_numbers(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.metadata["page"] for p in pages] == [1, 2, 3]
    assert "24 days of paid annual leave" in pages[0].page_content
    assert all(p.metadata["source"] == "sample.pdf" for p in pages)


def test_chunks_respect_size_and_have_ids(sample_pdf):
    chunks = ingest([sample_pdf], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3                       # pages were actually split
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all("page" in c.metadata for c in chunks)


def test_chunk_overlap_repeats_text(sample_pdf):
    chunks = chunk_documents(load_pdf(sample_pdf)[:1], chunk_size=200, chunk_overlap=80)
    first, second = chunks[0].page_content, chunks[1].page_content
    assert any(word in second for word in first.split()[-5:])
