"""
Reformulates a follow-up question as a standalone one using prior dialog.
Used before retrieval so anaphoric/elliptical queries ("А как его внедрить?")
become self-contained ("Как внедрить Scrum?") and match index content.
"""

import re

from llm_client import LLMClient


# Markers that suggest the question depends on prior turns (anaphora / ellipsis /
# meta follow-ups). When NONE are present and the question is reasonably long,
# it is almost certainly self-contained — skip the reformulation LLM call.
_ANAPHORA_RE = re.compile(
    r"\b(он|она|оно|они|его|её|ее|их|им|ими|ему|ей|нём|нем|ним|ней|неё|нее|"
    r"это|этот|эта|эти|этого|этом|этой|эту|этих|этим|тот|та|те|том|той|тех|"
    r"там|туда|оттуда|тогда|такой|такое|такие|таких|данн\w+)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"\b(подробнее|поподробнее|проще|короче|пример|примеры|ещё|еще|также|тоже|"
    r"перечисли|продолжи|разверни|поясни|уточни|сравни)\b",
    re.IGNORECASE,
)
_FOLLOWUP_PREFIX = ("а ", "и ", "но ", "ещё ", "еще ")

# Short answers — the reformulated query and the verdict are inherently brief.
_REFORMULATE_MAX_TOKENS = 64


def _needs_reformulation(question: str) -> bool:
    """Conservative gate: True unless the question is clearly self-contained.
    Biased toward reformulating (a wasted call costs little; a missed one hurts
    retrieval), so it only skips long questions with no referential markers."""
    q = question.strip()
    low = q.lower()
    if len(q) < 25:
        return True
    if low.startswith(_FOLLOWUP_PREFIX):
        return True
    return bool(_ANAPHORA_RE.search(low) or _META_RE.search(low))


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

        # Skip the LLM call entirely for clearly standalone questions.
        if not _needs_reformulation(question):
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
            raw = self.llm.complete(messages, max_tokens=_REFORMULATE_MAX_TOKENS)
        except Exception:
            return question

        if not raw:
            return question
        rewritten = raw.strip().strip('"').strip("«»").splitlines()[0].strip()
        return rewritten or question
