"""
Prompt construction for the RAG pipeline.
SYSTEM_PROMPT grounds answers in context and enforces citation.
build_messages formats retrieved chunks and optional dialog history.
"""

from chunker import Chunk


SYSTEM_PROMPT = (
    "Ты — ассистент для студентов по методологиям разработки проектов "
    "(Agile, Scrum, DevOps, технические задания).\n"
    "Отвечай на вопросы на основе предоставленного контекста. Используй контекст "
    "как основной источник: объединяй и структурируй информацию из нескольких "
    "фрагментов, если вопрос этого требует.\n"
    "Формулировка вопроса может не совпадать дословно с текстами — ищи смысловые "
    "соответствия, а не только точные совпадения слов.\n"
    "После каждого утверждения указывай номер источника в квадратных скобках: "
    "[1], [2] и т.д. — номера соответствуют пронумерованным фрагментам контекста.\n"
    "Если ни один фрагмент контекста не касается темы вопроса — честно напиши, "
    "что в предоставленных материалах нет ответа на этот вопрос, и не выдумывай факты.\n"
    "Отвечай на русском языке."
)


def build_messages(
    question: str,
    chunks: list[Chunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build GigaChat-compatible messages from the question, retrieved chunks,
    and optional dialog history. Chunks are numbered starting from 1.
    history: prior turns as [{"role": "user"|"assistant", "content": str}, ...].
    """
    numbered = [f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks)]
    context_block = "\n\n".join(numbered)

    user_content = (
        f"Контекст:\n{context_block}\n\n"
        f"Вопрос: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    return messages
