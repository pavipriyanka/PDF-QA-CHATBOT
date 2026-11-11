"""Evaluate retrieval quality: Hit Rate@k and MRR for vector-only, BM25-only and hybrid.

A question counts as a hit if a chunk from the correct page is in the top k.
MRR (Mean Reciprocal Rank) rewards putting the right chunk near the top.

Usage:  python scripts/evaluate.py [--pdf data/sample.pdf] [--eval data/eval_set.json] [--k 3]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pdfqa import PDFFile, RAGPipeline, settings  # noqa: E402
from pdfqa.models import get_embeddings  # noqa: E402


def score(rankings: list[list[int]], gold: list[int], k: int) -> dict:
    hits, rr = 0, 0.0
    for pages, page in zip(rankings, gold):
        top = pages[:k]
        if page in top:
            hits += 1
            rr += 1 / (top.index(page) + 1)
    return {"hit_rate": hits / len(gold), "mrr": rr / len(gold)}


def run(pdf_path: Path, eval_path: Path, k: int, embeddings=None) -> dict:
    pipeline = RAGPipeline(settings=settings, embeddings=embeddings or get_embeddings(settings))
    pipeline.index([PDFFile(pdf_path.name, pdf_path.read_bytes())], use_cache=False)
    r = pipeline.retriever
    chunks = {c.metadata["chunk_id"]: c for c in r.store.chunks}
    page_of = lambda ids: [chunks[i].metadata["page"] for i in ids]  # noqa: E731

    items = json.loads(eval_path.read_text())
    gold = [it["page"] for it in items]
    results = {}
    for name, rank in {
        "vector": lambda q: page_of(r._vector_ranking(q)),
        "bm25": lambda q: page_of(r._bm25_ranking(q)),
        "hybrid": lambda q: [d.metadata["page"] for d in r.retrieve(q, k=k)],
    }.items():
        results[name] = score([rank(it["question"]) for it in items], gold, k)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=ROOT / "data" / "sample.pdf")
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_set.json")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    results = run(args.pdf, args.eval, args.k)
    print(f"\nRetrieval evaluation (k={args.k}, {args.eval.name})")
    print(f"{'method':<8} {'hit_rate':>9} {'mrr':>6}")
    for name, m in results.items():
        print(f"{name:<8} {m['hit_rate']:>9.2f} {m['mrr']:>6.2f}")


if __name__ == "__main__":
    main()
