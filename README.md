<div align="center">

# 🧭 Свод — RAG-ассистент по проектному управлению

**Интеллектуальный помощник для студентов: отвечает на вопросы по Agile, Scrum, DevOps
и техническим заданиям, опираясь на выверенную базу знаний кафедры — со ссылками на источники.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![GigaChat](https://img.shields.io/badge/LLM-GigaChat-1A1A2E)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

</div>

---

## ✨ Возможности

- 🔍 **Гибридный поиск** по базе знаний: семантические эмбеддинги (e5) + BM25 + reranker.
- 📚 **Ответы с цитатами** — каждый факт сопровождается кликабельной ссылкой `[N]` на исходный фрагмент PDF.
- 💬 **Контекст диалога** — помнит ход беседы и переформулирует уточняющие вопросы.
- 🌗 **Светлая и тёмная темы** с переключателем и сохранением выбора.
- ⚡ **Плавный вывод ответа** (эффект печати), аккуратные анимации, адаптивная вёрстка для мобильных.
- 👍 **Обратная связь** по ответам и история бесед с авторизацией (включая гостевой режим).

---

## 🏗️ Архитектура

```
┌─────────────┐      /api      ┌──────────────┐    HTTP /ask   ┌──────────────────┐
│   frontend   │ ─────────────▶ │ web_backend  │ ─────────────▶ │   rag_service     │
│ React + Vite │  (nginx proxy) │  FastAPI BFF │                │ FastAPI + RAG      │
│   :8090      │ ◀───────────── │   :8001      │ ◀───────────── │   :8000            │
└─────────────┘                 └──────┬───────┘                └─────────┬──────────┘
                                       │                                  │
                                  SQLite (история,                 Эмбеддинги (e5) +
                                  пользователи)                    BM25 + reranker +
                                                                   GigaChat (LLM)
```

| Сервис        | Технологии                          | Порт (хост) | Назначение                                  |
|---------------|-------------------------------------|:-----------:|---------------------------------------------|
| `frontend`    | React, Vite, TailwindCSS, nginx     | **8090**    | Веб-интерфейс (SPA), проксирует `/api`      |
| `web_backend` | FastAPI, SQLite, JWT                 | **8001**    | Авторизация, история, обратная связь        |
| `rag_service` | FastAPI, sentence-transformers, GigaChat | **8000** | Поиск по базе знаний и генерация ответа      |

---

## 🧰 Технологический стек

- **Frontend:** React + TypeScript, Vite, TailwindCSS, react-markdown.
- **Backend (BFF):** FastAPI, SQLAlchemy, SQLite, JWT-аутентификация.
- **RAG-сервис:** FastAPI, PyMuPDF (извлечение текста PDF), `intfloat/multilingual-e5-base`
  (эмбеддинги), BM25, `BAAI/bge-reranker-v2-m3` (reranker), GigaChat (LLM).
- **Инфраструктура:** Docker Compose.

---

## 📁 Структура проекта

```
AI-Assistant/
├── docker-compose.yml          # оркестрация всех трёх сервисов
├── frontend/                   # React + Vite SPA
├── web/backend/                # FastAPI BFF (auth, история, обратная связь)
└── rag_service/                # RAG-пайплайн и HTTP API
    ├── api.py                  # FastAPI приложение (/ask, /documents, /health)
    ├── knowledge_base/         # PDF-методички (источники)
    ├── data/                   # собранный индекс (генерируется, в .gitignore)
    ├── scripts/                # build_chunks.py, build_index.py и др.
    └── src/                    # загрузчики, чанкер, эмбеддер, retriever, пайплайн
```

---

## ✅ Предварительные требования

- **Docker** и **Docker Compose** (v2) — для запуска.
- **Python 3.11+** — для одноразовой сборки индекса базы знаний.
- **Node.js 20+** — только если запускаете фронтенд в режиме разработки.
- **Ключ авторизации GigaChat** — получите в личном кабинете
  [developers.sber.ru](https://developers.sber.ru/portal/products/gigachat).

---

## 🔐 Шаг 1. Создание файлов `.env`

> Файлы `.env` не хранятся в репозитории — их нужно создать вручную.

**`rag_service/.env`**
```env
# Ключ авторизации GigaChat (Base64 из client_id:client_secret)
GIGACHAT_AUTH_KEY=ваш-ключ-gigachat

# (опционально) токен Hugging Face, если упираетесь в лимиты загрузки моделей
HF_TOKEN=
```

**`web/backend/.env`**
```env
# Секрет для подписи JWT. Сгенерируйте: openssl rand -hex 32
JWT_SECRET=замените-на-случайную-строку-32-байта
```

---

## 📦 Шаг 2. Сборка индекса базы знаний (однократно)

`rag_service` при старте требует готовый индекс в `rag_service/data/`. Каталог монтируется
в контейнер **только для чтения**, поэтому индекс нужно собрать на хосте заранее.

```bash
cd rag_service

# создаём виртуальное окружение и ставим зависимости
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# собираем чанки из PDF и строим векторный индекс
python scripts/build_chunks.py
python scripts/build_index.py
```

> ⏳ При первом запуске `build_index.py` скачает модели эмбеддингов и reranker
> (≈1–2 ГБ) и проиндексирует базу знаний. Это может занять несколько минут.
> Результат — файлы `data/vectors.npy`, `data/chunks_meta.json`, `data/chunks.jsonl`.

---

## 🚀 Шаг 3. Запуск через Docker Compose (рекомендуется)

Из корня проекта:

```bash
docker compose up --build
```

Дождитесь, пока `rag_service` прогрузит модели (см. healthcheck, ~1–2 минуты), и откройте:

### 👉 http://localhost:8090

Остановка:
```bash
docker compose down
```

---

## 🛠️ Альтернатива: запуск для разработки (без Docker)

Откройте **три терминала** (индекс из Шага 2 должен быть уже собран).

**1. RAG-сервис** (порт 8000)
```bash
cd rag_service
source venv/bin/activate        # Windows: venv\Scripts\activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

**2. Web-backend** (порт 8001)
```bash
cd web/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

**3. Frontend** (порт 5173, Vite проксирует `/api` на :8001)
```bash
cd frontend
npm install
npm run dev
```

Откройте http://localhost:5173

> В dev-режиме действуют значения по умолчанию: backend обращается к RAG по
> `http://localhost:8000`, база данных — локальный файл `web.db`, CORS разрешён для `:5173`.

---

## 🌐 Карта портов

| URL                       | Сервис        | Когда                |
|---------------------------|---------------|----------------------|
| http://localhost:8090     | Frontend      | Docker Compose       |
| http://localhost:5173     | Frontend      | Режим разработки     |
| http://localhost:8001/api | Web-backend   | всегда               |
| http://localhost:8000     | RAG-сервис    | всегда               |
| http://localhost:8000/health | Health-check RAG | диагностика     |

---

## 🔄 Обновление базы знаний

Чтобы добавить или заменить методички:

1. Положите PDF в `rag_service/knowledge_base/`.
2. Пересоберите индекс:
   ```bash
   cd rag_service && source venv/bin/activate
   python scripts/build_chunks.py && python scripts/build_index.py
   ```
3. Перечитайте индекс сервисом:
   ```bash
   docker compose restart rag_service
   ```

---

## ⚙️ Полезные переменные окружения

Задаются в `docker-compose.yml` (RAG) или в `.env` (backend). Значения по умолчанию подходят
для большинства сценариев.

| Переменная             | По умолчанию | Описание                                        |
|------------------------|:------------:|-------------------------------------------------|
| `RAG_TOP_K`            | `5`          | сколько фрагментов передаётся в LLM             |
| `RAG_SCORE_THRESHOLD`  | `0.78`       | порог релевантности (отсекает офтоп)            |
| `GIGACHAT_MAX_TOKENS`  | `1024`       | ограничение длины ответа                        |
| `GUEST_QUERY_LIMIT`    | `5`          | лимит запросов для гостевого аккаунта           |

---

## 🩺 Troubleshooting

- **`rag_service` падает при старте с «Index file missing»** — не собран индекс. Выполните Шаг 2.
- **`GIGACHAT_AUTH_KEY is not set`** — не создан `rag_service/.env` или ключ пустой.
- **Долгий первый запуск** — скачиваются модели (e5 + reranker). Кэш сохраняется в
  docker-том `hf_cache`, повторные запуски быстрые.
- **502 / «нет соединения с базой знаний» в интерфейсе** — `rag_service` ещё прогревается;
  проверьте http://localhost:8000/health.

---

## 📋 Управление проектом

Доска Yougile: https://en.yougile.com/board/uvsbei7v43nx
