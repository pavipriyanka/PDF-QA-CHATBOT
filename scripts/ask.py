"""Command-line chat with your PDFs (no UI needed).

Usage:  python scripts/ask.py data/sample.pdf [more.pdf ...]
Type a question and press Enter; 'exit' to quit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdfqa import PDFFile, RAGPipeline, settings  # noqa: E402
from pdfqa.models import get_embeddings, get_llm  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pdfs = [PDFFile(Path(p).name, Path(p).read_bytes()) for p in sys.argv[1:]]
    pipeline = RAGPipeline(settings=settings, embeddings=get_embeddings(settings), llm=get_llm(settings))
    print(f"Indexed {pipeline.index(pdfs)} chunks. Ask away ('exit' to quit).")
    if pipeline.llm is None:
        print("(No GOOGLE_API_KEY set: showing relevant passages instead of generated answers.)")

    while (question := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        if not question:
            continue
        result = pipeline.ask(question)
        print("Bot: ", end="")
        answer = ""
        for token in result.stream:
            print(token, end="", flush=True)
            answer += token
        print("\nSources: " + ", ".join(
            f"[{i}] {d.metadata['source']} p.{d.metadata['page']}" for i, d in enumerate(result.sources, 1)))
        pipeline.remember(question, answer)


if __name__ == "__main__":
    main()
