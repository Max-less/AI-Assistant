import time
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole


class LLMClient:
    """GigaChat API client with retry and exponential backoff."""

    def __init__(self, auth_key: str, timeout: int = 30, max_retries: int = 3):
        self.auth_key = auth_key
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages: list[dict]) -> str:
        """
        Send messages to GigaChat and return the response text.

        messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        Retry: 3 attempts with exponential backoff (1s, 2s, 4s).
        """
        giga_messages = [
            Messages(role=MessagesRole(m["role"]), content=m["content"])
            for m in messages
        ]
        payload = Chat(messages=giga_messages)

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                with GigaChat(
                    credentials=self.auth_key,
                    verify_ssl_certs=False,
                    timeout=self.timeout,
                ) as giga:
                    response = giga.chat(payload)
                    return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(delay)

        raise last_exception
