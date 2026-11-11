# 📄 PDF Q&A Chatbot: Hybrid RAG with Citations

Chat with one or more PDFs. Answers are grounded in your documents and cite the exact file and page. Retrieval combines **BM25 keyword search** and **FAISS semantic search**, fused with **Reciprocal Rank Fusion**. Follow-up questions work thanks to **conversational query rewriting**.

## Features
- 📚 **Multi-PDF upload** with page-level citations `[1] handbook.pdf · page 3`
- 🔀 **Hybrid retrieval**: BM25 + dense vectors + RRF fusion
- 💬 **Conversational memory**: follow-ups like *"and for interns?"* are rewritten into standalone questions
- ⚡ **Streaming answers** token by token
- 💾 **Index caching**: re-uploading the same PDFs loads the saved FAISS index instead of re-embedding
- 🛡️ **Grounded prompting**: answers only from sources, and says so when the answer isn't there
- 🔌 **Works without an API key**: falls back to showing the most relevant passages
- 📊 **Retrieval evaluation script**: Hit Rate@k and MRR for vector vs BM25 vs hybrid
- ✅ **15 unit tests** that run offline with fake embeddings and a fake LLM

## Architecture
```
            ┌──────────── Indexing ────────────┐
 PDFs ──► pypdf (per page) ──► RecursiveCharacterTextSplitter ──► chunks + metadata
                                                                     │
                                         ┌───────────────────────────┴──────────┐
                                         ▼                                      ▼
                          HuggingFace embeddings → FAISS (cosine)        BM25 index
            ┌──────────── Question ────────────┐
 question + chat history ──► LLM rewrites to standalone question
                         ──► FAISS top-20  ┐
                         ──► BM25  top-20  ┴─► Reciprocal Rank Fusion ─► top-5 chunks
                         ──► Gemini with numbered sources ─► streamed answer with [n] citations
```

## Project structure
```
pdf-qa-chatbot/
├── app.py                  # Streamlit UI
├── src/pdfqa/
│   ├── config.py           # settings from .env
│   ├── ingest.py           # PDF loading + chunking
│   ├── vector_store.py     # FAISS wrapper with save/load
│   ├── retriever.py        # BM25 + FAISS + Reciprocal Rank Fusion
│   ├── prompts.py          # condense + answer prompts
│   ├── models.py           # embedding model + Gemini LLM factories
│   └── pipeline.py         # RAGPipeline: index → condense → retrieve → answer
├── scripts/
│   ├── ask.py              # chat from the terminal
│   ├── evaluate.py         # retrieval metrics
│   └── make_sample_pdf.py  # regenerates data/sample.pdf
├── data/
│   ├── sample.pdf          # demo document (company handbook)
│   └── eval_set.json       # 12 questions with the correct page
├── tests/                  # pytest suite (offline)
├── requirements.txt / requirements-dev.txt
├── .env.example
└── Dockerfile
```

## Quick start
Requires **Python 3.11 or 3.12** (3.10+ also works).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your GOOGLE_API_KEY into .env
streamlit run app.py
```
Open http://localhost:8501, upload `data/sample.pdf` (or your own PDFs), click **Process documents** and ask:
- *How much annual leave do I get?* → then *and sick leave?*
- *Which ticket code do I use for a broken laptop?*

The first run downloads the embedding model (~90 MB) once.

### Terminal chat
```bash
python scripts/ask.py data/sample.pdf
```

### Evaluate retrieval
```bash
python scripts/evaluate.py --k 3
```
Prints Hit Rate@3 and MRR for vector-only, BM25-only and hybrid retrieval on `data/eval_set.json`.

### Run tests
```bash
pip install -r requirements-dev.txt
pytest
```

### Docker
```bash
docker build -t pdf-qa .
docker run -p 8501:8501 --env-file .env pdf-qa
```

## Key design decisions
| Decision | Why |
|---|---|
| Chunk size 1000, overlap 150 | Big enough to hold a full idea; overlap stops facts being cut at chunk boundaries |
| `all-MiniLM-L6-v2` embeddings | Free, runs locally on CPU, 384-dim, strong quality for its size |
| Normalised vectors + `IndexFlatIP` | Inner product on unit vectors = cosine similarity; exact search is fast enough for thousands of chunks |
| Hybrid BM25 + vectors | Vectors handle paraphrases; BM25 handles exact terms (codes, names, numbers) |
| Reciprocal Rank Fusion (k=60) | Merges rankings without calibrating BM25 and cosine scores, which use different scales |
| Query rewriting with history | Retrieval only sees the current question; "and sick leave?" alone retrieves nothing useful |
| Numbered sources + "only use sources" rule | Lets users verify answers and reduces hallucination |
| Own FAISS wrapper storing chunks as JSON | No pickle (safe to load), no dependency on the retired `langchain-community` |
| Cache key = hash(PDF bytes + model + chunk settings) | Changing any setting automatically rebuilds the index |

## Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | – | Gemini key (free at aistudio.google.com/apikey) |
| `LLM_MODEL` | `gemini-2.5-flash` | Chat model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Characters |
| `TOP_K` | 5 | Chunks passed to the LLM |
| `CANDIDATE_K` | 20 | Chunks fetched from each retriever before fusion |
| `HISTORY_TURNS` | 3 | Past turns used for query rewriting |

## Future improvements
- Cross-encoder re-ranking (e.g. `bge-reranker`) after fusion
- OCR (Tesseract) for scanned PDFs
- Answer-quality evaluation (faithfulness, relevance) with RAGAS
- Persistent multi-user storage (Qdrant / pgvector)

## Tech stack
Python · LangChain (core, text-splitters, HuggingFace, Google GenAI) · FAISS · rank-bm25 · sentence-transformers · Google Gemini · Streamlit · pytest · Docker
