FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

COPY rag_service/requirements.txt /app/rag_service/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/rag_service/requirements.txt

COPY rag_service /app/rag_service

EXPOSE 8000

CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
