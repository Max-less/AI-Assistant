import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retriever import fuse_rrf


def test_fuse_doc_in_both_lists_sums_weights():
    fused = fuse_rrf(["a", "b"], ["a", "c"], alpha=0.5, k=60)
    scores = dict(fused)
    # "a" is rank 1 in both → 0.5/61 + 0.5/61
    assert scores["a"] == 1.0 / 61
    # "b" only in dense at rank 2 → 0.5/62
    assert scores["b"] == 0.5 / 62
    # "c" only in bm25 at rank 2 → 0.5/62
    assert scores["c"] == 0.5 / 62


def test_fuse_alpha_one_is_dense_only():
    """alpha=1.0 → BM25 ranking ignored, dense ranking preserved."""
    fused = fuse_rrf(["a", "b", "c"], ["c", "b", "a"], alpha=1.0, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b", "c"]
    # bm25-only doc would get score 0 — but here all docs appear in dense
    scores = dict(fused)
    assert scores["a"] == 1.0 / 61


def test_fuse_alpha_zero_is_bm25_only():
    """alpha=0.0 → dense ranking ignored, bm25 ranking preserved."""
    fused = fuse_rrf(["a", "b", "c"], ["c", "b", "a"], alpha=0.0, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["c", "b", "a"]


def test_fuse_output_sorted_descending():
    fused = fuse_rrf(["a", "b", "c"], ["b", "c", "a"], alpha=0.5, k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_fuse_doc_only_in_one_list():
    fused = fuse_rrf(["a"], ["b"], alpha=0.5, k=60)
    scores = dict(fused)
    assert set(scores.keys()) == {"a", "b"}
    # equal rank → equal score
    assert scores["a"] == scores["b"]


def test_fuse_empty_lists():
    assert fuse_rrf([], [], alpha=0.5, k=60) == []


def test_fuse_one_empty_list():
    fused = fuse_rrf(["a", "b"], [], alpha=0.5, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b"]
    # only dense contribution, scaled by alpha
    scores = dict(fused)
    assert scores["a"] == 0.5 / 61
