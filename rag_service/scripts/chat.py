"""
CLI chat with GigaChat.
Run: python scripts/chat.py (from rag_service/ folder, with active venv)
Exit: quit, exit or Ctrl+C
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from llm_client import LLMClient

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def main():
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key or auth_key == "your-authorization-key-here":
        print("ERROR: Set GIGACHAT_AUTH_KEY in .env file")
        sys.exit(1)

    client = LLMClient(auth_key)
    messages = []

    print("GigaChat CLI (quit/exit to leave)")
    print("-" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.complete(messages)
            print(f"Bot: {response}")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}")
            messages.pop()


if __name__ == "__main__":
    main()
