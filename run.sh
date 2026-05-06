#!/usr/bin/env bash
# Starts the Phi-2 Serving API on http://0.0.0.0:8000 with hot reload.
# The model is loaded once at app startup via FastAPI's lifespan event;
# subsequent requests reuse the in-memory singleton.
set -euo pipefail
cd "$(dirname "$0")"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
