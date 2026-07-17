# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL org.opencontainers.image.title="phi2-serving" \
      org.opencontainers.image.description="FastAPI + PyTorch CPU serving API for Phi-2 (base model; MLX LoRA adapter not applied in-container)" \
      org.opencontainers.image.version="0.1.0"

WORKDIR /app

COPY requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements-linux.txt

COPY app/ ./app/

RUN mkdir -p /app/logs

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main_torch:app", "--host", "0.0.0.0", "--port", "8000"]
