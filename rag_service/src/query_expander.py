"""
Query decomposition for multi-topic questions.
Uses the LLM to split a compound question into focused sub-queries.
Single-topic questions pass through unchanged.
"""

from llm_client import LLMClient


EXPANDER_SYSTEM_PROMPT = (
    "Ты помогаешь улучшить поиск по базе знаний. "
    "Получив вопрос пользователя, реши: содержит ли он несколько самостоятельных тем?\n"
    "Если ДА — разбей вопрос на 2–4 коротких независимых поисковых запроса, "
    "каждый на отдельной строке, без нумерации и без маркеров.\n"
    "Если НЕТ — верни исходный вопрос ровно одной строкой.\n"
    "Не добавляй объяснений, комментариев или заголовков — только сами запросы."
)


class QueryExpander:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def expand(self, question: str) -> list[str]:
        """Return 1–4 focused sub-queries. Falls back to [question] on any error."""
        messages = [
            {"role": "system", "content": EXPANDER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        try:
            raw = self.llm.complete(messages)
        except Exception:
            return [question]

        lines = [line.strip(" -•*\t") for line in raw.splitlines()]
        queries = [line for line in lines if line]

        if not queries:
            return [question]

        return queries[:4]
