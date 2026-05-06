"""FastAPI app for serving the fine-tuned Phi-2 model on the PyTorch backend.

Identical to `app.main` except it imports `app.model_torch` (HuggingFace
Transformers + PyTorch) instead of `app.model` (MLX). Use this entry
point inside Linux containers where MLX is not available.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import model_torch as model
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "requests.log"

logger = logging.getLogger("phi2-serving")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; cleanup hook is a no-op for now."""
    model.load_model()
    yield


app = FastAPI(title="Phi-2 Serving API (PyTorch)", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log endpoint, prompt preview, response time, and status code."""
    start = time.perf_counter()
    prompt_preview = ""
    if request.method == "POST" and request.url.path.startswith("/generate"):
        try:
            body = await request.body()
            request._body = body
            import json

            data = json.loads(body or b"{}")
            prompt_preview = str(data.get("prompt", ""))[:100]
        except Exception:
            prompt_preview = ""

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "endpoint=%s prompt=%r response_time_ms=%.2f status=%d",
        request.url.path,
        prompt_preview,
        elapsed_ms,
        response.status_code,
    )
    return response


def _require_loaded() -> None:
    """Raise 503 if the model isn't ready to serve."""
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe with model-loaded flag."""
    return HealthResponse(status="ok", model_loaded=model.is_loaded())


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return static info about the currently loaded model."""
    _require_loaded()
    return ModelInfo(**model.get_info())


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Run synchronous generation and return the full completion."""
    _require_loaded()
    text = model.generate(req.prompt, req.max_tokens)
    tokens_used = len(text.split())
    return GenerateResponse(response=text, prompt=req.prompt, tokens_used=tokens_used)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Stream tokens to the client, newline-separated, as they are produced."""
    _require_loaded()

    def iterator():
        for token in model.stream_generate(req.prompt, req.max_tokens):
            yield token + "\n"

    return StreamingResponse(iterator(), media_type="text/plain")
