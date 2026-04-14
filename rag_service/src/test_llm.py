import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()

auth_key = os.getenv("GIGACHAT_AUTH_KEY")

if not auth_key or auth_key == "your-authorization-key-here":
    print("ERROR: Set GIGACHAT_AUTH_KEY in .env file")
    exit(1)

print("Connecting to GigaChat...")

try:
    with GigaChat(credentials=auth_key, verify_ssl_certs=False) as giga:
        response = giga.chat("Привет! Кратко объясни, что такое Scrum.")
        print()
        print("GigaChat response:")
        print(response.choices[0].message.content)
except Exception as e:
    print(f"ERROR calling GigaChat API: {e}")
    exit(1)