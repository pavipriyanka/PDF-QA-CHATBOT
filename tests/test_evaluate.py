import json

import evaluate


def test_score_metrics():
    m = evaluate.score([[1, 2], [3, 1], [2, 3]], gold=[1, 1, 1], k=2)
    assert m["hit_rate"] == 2 / 3
    assert m["mrr"] == (1 + 0.5) / 3


def test_evaluate_runs_end_to_end(tmp_path, sample_pdf, fake_embeddings):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(sample_pdf.data)
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps([{"question": "ticket code IT-HW-07", "page": 2}]))
    results = evaluate.run(pdf, ev, k=3, embeddings=fake_embeddings)
    assert set(results) == {"vector", "bm25", "hybrid"}
    assert results["bm25"]["hit_rate"] == 1.0
