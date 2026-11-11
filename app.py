"""Streamlit UI for the PDF Q&A chatbot.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdfqa import PDFFile, RAGPipeline, settings  # noqa: E402
from pdfqa.models import get_embeddings, get_llm  # noqa: E402

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_embeddings():
    return get_embeddings(settings)


def sidebar():
    with st.sidebar:
        st.header("⚙️ Setup")
        api_key = st.text_input(
            "Gemini API key", value=settings.google_api_key, type="password",
            help="Free key: https://aistudio.google.com/apikey. Without a key the app "
                 "shows the most relevant passages instead of a generated answer.",
        )
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        process = st.button("Process documents", type="primary", disabled=not files)
        if st.session_state.get("indexed"):
            st.success(st.session_state.indexed)
            if st.button("Clear chat"):
                st.session_state.messages = []
                st.session_state.pipeline.history.clear()
                st.rerun()
    return api_key, files, process


def index_documents(files, api_key):
    pdfs = [PDFFile(name=f.name, data=f.getvalue()) for f in files]
    pipeline = RAGPipeline(settings=settings, embeddings=load_embeddings(), llm=get_llm(settings, api_key))
    with st.spinner("Reading, chunking and indexing..."):
        n_chunks = pipeline.index(pdfs)
    st.session_state.update(
        pipeline=pipeline, messages=[],
        indexed=f"Indexed {len(pdfs)} PDF(s) → {n_chunks} chunks",
    )


def render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, d in enumerate(sources, start=1):
            m = d.metadata
            st.markdown(f"**[{i}] {m['source']} · page {m['page']}** · relevance {m['score']}")
            st.caption(d.page_content[:500] + ("…" if len(d.page_content) > 500 else ""))


def main():
    st.title("📄 PDF Q&A Chatbot")
    st.caption("Hybrid search (BM25 + FAISS) · conversational memory · cited answers")

    api_key, files, process = sidebar()
    if process:
        try:
            index_documents(files, api_key)
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    if "pipeline" not in st.session_state:
        st.info("👈 Upload one or more PDFs and click **Process documents** to start.")
        return

    pipeline: RAGPipeline = st.session_state.pipeline
    pipeline.llm = get_llm(settings, api_key)   # pick up a key entered after indexing

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        try:
            result = pipeline.ask(question)
            if result.question != question:
                st.caption(f"🔎 Searched for: *{result.question}*")
            answer = st.write_stream(result.stream)
        except Exception as e:  # bad key, quota exceeded, network errors...
            st.error(f"LLM request failed: {e}")
            return
        render_sources(result.sources)

    pipeline.remember(question, answer)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": result.sources},
    ]


main()
