"""
Reformulates a follow-up question as a standalone one using prior dialog.
Used before retrieval so anaphoric/elliptical queries ("А как его внедрить?")
become self-contained ("Как внедрить Scrum?") and match index content.
"""

from llm_client import LLMClient


REFORMULATE_SYSTEM_PROMPT = (
    "Ты помогаешь улучшить поиск по базе знаний в многоходовом диалоге.\n"
    "Получив историю диалога и новый вопрос пользователя, перепиши его как "
    "самостоятельный вопрос, понятный без контекста: подставь явные сущности "
    "вместо местоимений («его», «это», «там»), раскрой эллипсисы.\n"
    "Если новый вопрос уже самостоятельный — верни его без изменений.\n"
    "Верни только сам переформулированный вопрос одной строкой, без объяснений, "
    "без кавычек, без префиксов вроде «Вопрос:»."
)


class DialogManager:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reformulate(self, question: str, history: list[dict]) -> str:
        """Return a standalone version of `question` given prior turns.
        Falls back to the original question on any LLM error or empty response."""
        if not history:
            return question

        history_block = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
        user_content = (
            f"История диалога:\n{history_block}\n\n"
            f"Новый вопрос: {question}\n\n"
            f"Переформулированный самостоятельный вопрос:"
        )
        messages = [
            {"role": "system", "content": REFORMULATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = self.llm.complete(messages)
        except Exception:
            return question

        if not raw:
            return question
        rewritten = raw.strip().strip('"').strip("«»").splitlines()[0].strip()
        return rewritten or question
