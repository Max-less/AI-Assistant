import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bm25 import BM25, tokenize


def test_tokenize_cyrillic_and_latin():
    assert tokenize("Что такое Scrum?") == ["что", "такое", "scrum"]
    assert tokenize("DevOps — это культура.") == ["devops", "это", "культура"]
    assert tokenize("") == []


def test_search_ranks_exact_match_first():
    corpus = [
        "Scrum это методология управления проектами в гибкой разработке",
        "DevOps объединяет разработку и эксплуатацию программного обеспечения",
        "Техническое задание описывает требования заказчика к системе",
        "Канбан использует визуализацию задач на доске",
        "Водопадная модель предполагает последовательное выполнение этапов",
    ]
    bm25 = BM25().fit(corpus)
    results = bm25.search("что такое Scrum", top_k=3)

    assert len(results) == 3
    assert results[0][0] == 0
    assert results[0][1] > 0


def test_search_returns_top_k_sorted():
    corpus = [f"документ {i} содержит уникальное слово token{i}" for i in range(6)]
    bm25 = BM25().fit(corpus)
    results = bm25.search("token2 token4", top_k=2)

    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    # both matching docs should be on top
    matched_indices = {idx for idx, _ in results}
    assert matched_indices == {2, 4}


def test_search_unknown_term_returns_zero_scores():
    corpus = ["Scrum methodology", "DevOps practices", "Kanban board"]
    bm25 = BM25().fit(corpus)
    results = bm25.search("совершенно неизвестный термин", top_k=2)

    assert len(results) == 2
    for _, score in results:
        assert score == 0.0


def test_idf_penalizes_common_terms():
    # "общий" appears in every doc; "редкий" only in doc 1
    corpus = [
        "общий текст про методологию",
        "общий текст и редкий термин здесь",
        "общий документ совсем другой",
    ]
    bm25 = BM25().fit(corpus)

    # "общий" is in all docs → IDF should be ~0 (log(1 + 0.5/3.5) is small)
    common_idf = bm25.idf["общий"]
    rare_idf = bm25.idf["редкий"]
    assert rare_idf > common_idf
    assert common_idf < 0.5

    # Query mixes common + rare → doc with rare term must rank first
    results = bm25.search("общий редкий", top_k=3)
    assert results[0][0] == 1


def test_empty_query_returns_empty():
    bm25 = BM25().fit(["some doc", "another doc"])
    assert bm25.search("") == []
    assert bm25.search("   ") == []


def test_empty_corpus_returns_empty():
    bm25 = BM25().fit([])
    assert bm25.search("anything", top_k=5) == []
