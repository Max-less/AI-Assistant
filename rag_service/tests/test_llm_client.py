import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from llm_client import LLMClient

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def test_complete_returns_correct_answer():
    """Smoke-тест: GigaChat должен ответить что 2+2=4."""
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    assert auth_key, "GIGACHAT_AUTH_KEY не задан в .env"

    client = LLMClient(auth_key)
    response = client.complete([
        {"role": "user", "content": "Сколько будет 2+2? Ответь только числом."}
    ])

    assert "4" in response, f"Ожидалось '4' в ответе, получено: {response}"
