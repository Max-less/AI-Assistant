import os
import time

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole


class LLMClient:
    """GigaChat API client with retry and exponential backoff.
    Holds a single persistent GigaChat instance to avoid re-authenticating
    and re-establishing HTTP sessions on every call."""

    def __init__(
        self,
        auth_key: str,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        self.auth_key = auth_key
        self.timeout = (
            timeout if timeout is not None else int(os.getenv("GIGACHAT_TIMEOUT", "30"))
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("GIGACHAT_MAX_RETRIES", "3"))
        )
        self._giga = GigaChat(
            credentials=self.auth_key,
            verify_ssl_certs=False,
            timeout=self.timeout,
        )

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """
        Send messages to GigaChat and return the response text.

        messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        max_tokens: optional cap on the generated answer length (saves output
        tokens; keep generous for answers, small for short auxiliary calls).
        Retries with exponential backoff (1s, 2s, 4s, ...).
        """
        giga_messages = [
            Messages(role=MessagesRole(m["role"]), content=m["content"])
            for m in messages
        ]
        payload = Chat(messages=giga_messages)
        if max_tokens is not None:
            payload.max_tokens = max_tokens

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._giga.chat(payload)
                return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    time.sleep(delay)

        raise last_exception

    def close(self) -> None:
        """Release the persistent GigaChat HTTP session."""
        close = getattr(self._giga, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
