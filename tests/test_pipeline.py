import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from pdfqa import PDFFile, RAGPipeline
from pdfqa.pipeline import format_sources


def make_pipeline(test_settings, fake_embeddings, llm=None):
    return RAGPipeline(settings=test_settings, embeddings=fake_embeddings, llm=llm)


def test_ask_before_index_raises(test_settings, fake_embeddings):
    with pytest.raises(RuntimeError):
        make_pipeline(test_settings, fake_embeddings).ask("hi")


def test_empty_pdf_raises(test_settings, fake_embeddings):
    from reportlab.pdfgen import canvas
    import io

    buf = io.BytesIO()
    canvas.Canvas(buf).save()  # one blank page, no text
    with pytest.raises(ValueError, match="No text found"):
        make_pipeline(test_settings, fake_embeddings).index([PDFFile("blank.pdf", buf.getvalue())])


def test_extractive_mode_without_llm(sample_pdf, test_settings, fake_embeddings):
    p = make_pipeline(test_settings, fake_embeddings)
    p.index([sample_pdf])
    result = p.ask("What is the home office allowance?")
    text = "".join(result.stream)
    assert "No API key" in text
    assert len(result.sources) == test_settings.top_k


def test_llm_answer_streams_and_condenses(sample_pdf, test_settings, fake_embeddings):
    llm = FakeListChatModel(responses=[
        "Employees get 24 days of annual leave [1].",           # answer to Q1
        "How many days of sick leave do employees get?",         # condensed Q2
        "12 days of sick leave [1].",                            # answer to Q2
    ])
    p = make_pipeline(test_settings, fake_embeddings, llm)
    p.index([sample_pdf])

    r1 = p.ask("How much annual leave do I get?")
    a1 = "".join(r1.stream)
    assert a1.startswith("Employees get 24 days")
    p.remember("How much annual leave do I get?", a1)

    r2 = p.ask("and sick leave?")
    assert r2.question == "How many days of sick leave do employees get?"
    assert "".join(r2.stream) == "12 days of sick leave [1]."


def test_index_uses_cache(sample_pdf, test_settings, fake_embeddings):
    p = make_pipeline(test_settings, fake_embeddings)
    n = p.index([sample_pdf])
    assert any(test_settings.cache_dir.iterdir())
    assert make_pipeline(test_settings, fake_embeddings).index([sample_pdf]) == n


def test_format_sources_numbers_and_pages(sample_pdf, test_settings, fake_embeddings):
    p = make_pipeline(test_settings, fake_embeddings)
    p.index([sample_pdf])
    text = format_sources(p.retriever.retrieve("probation period", k=2))
    assert text.startswith("[1] (sample.pdf, page ")
    assert "\n\n[2] (" in text
