"""Quick dense top-1 cosine for ad-hoc questions (embed-only, no rerank => fast).
Used to sanity-check the relevance gate against realistic SHORT phrasings."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedder import Embedder
from vector_store import VectorStore

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

IN_BASE = [
    "Что такое Scrum?",
    "Что такое DevOps?",
    "Что такое CI/CD?",
    "Что такое спринт?",
    "Кто такой Scrum-мастер?",
    "Что такое Kanban?",
    "Что такое бэклог продукта?",
    "Что такое техническое задание?",
    "Что такое ретроспектива?",
    "Что такое жизненный цикл проекта?",
]
OUT_BASE = [
    "Какая завтра погода?",
    "Кто написал Войну и мир?",
    "Как установить Python?",
    "Как написать SQL JOIN?",
    "Посоветуй фильмы на вечер.",
]


def main() -> None:
    store = VectorStore.load_with_texts(
        os.path.join(DATA, "vectors.npy"),
        os.path.join(DATA, "chunks_meta.json"),
        os.path.join(DATA, "chunks.jsonl"),
    )
    emb = Embedder()

    def top1(q: str) -> float:
        res = store.search(emb.embed_query(q), top_k=1)
        return res[0][1] if res else 0.0

    print("IN-BASE (must NOT be gated):")
    for q in IN_BASE:
        print(f"  {top1(q):.3f}  {q}")
    print("OUT-OF-BASE (should be gated):")
    for q in OUT_BASE:
        print(f"  {top1(q):.3f}  {q}")


if __name__ == "__main__":
    main()
