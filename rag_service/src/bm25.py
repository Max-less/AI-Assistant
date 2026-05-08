"""
BM25 ranking on numpy.
Okapi BM25 with +1 IDF smoothing; Unicode-aware tokenization for Russian + English.
"""

import math
import re
from collections import Counter

import numpy as np


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_lens: np.ndarray = np.zeros(0, dtype=np.float64)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> "BM25":
        self.doc_count = len(corpus)
        token_lists = [tokenize(doc) for doc in corpus]
        self.doc_lens = np.array([len(t) for t in token_lists], dtype=np.float64)
        self.avgdl = float(self.doc_lens.mean()) if self.doc_count > 0 else 0.0

        self.postings = {}
        df: dict[str, int] = {}
        for doc_idx, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((doc_idx, freq))
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log(1.0 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for term, doc_freq in df.items()
        }
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self.doc_count == 0:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float64)
        # length-normalization factor per doc; avgdl=0 only when corpus is empty (handled above)
        norm = 1.0 - self.b + self.b * (self.doc_lens / self.avgdl)

        for term in tokens:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_idx, freq in postings:
                denom = freq + self.k1 * norm[doc_idx]
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / denom

        top_k = min(top_k, self.doc_count)
        if top_k == 0:
            return []
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in top_indices]
