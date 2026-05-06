# syntax=docker/dockerfile:1
FROM --platform=linux/arm64 python:3.11-slim

LABEL org.opencontainers.image.title="phi2-serving" \
      org.opencontainers.image.description="FastAPI + MLX serving API for a fine-tuned Phi-2 model" \
      org.opencontainers.image.version="0.1.0"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY run.sh ./run.sh

RUN mkdir -p /app/logs

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
