import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()

auth_key = os.getenv("GIGACHAT_AUTH_KEY")

if not auth_key or auth_key == "your-authorization-key-here":
    print("ОШИБКА: Укажите GIGACHAT_AUTH_KEY в файле .env")
    exit(1)

print("Подключаюсь к GigaChat...")

try:
    with GigaChat(credentials=auth_key, verify_ssl_certs=False) as giga:
        response = giga.chat("Привет! Кратко объясни, что такое Scrum.")
        print()
        print("Ответ GigaChat:")
        print(response.choices[0].message.content)
except Exception as e:
    print(f"ОШИБКА при обращении к GigaChat API: {e}")
    exit(1)