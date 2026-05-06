"""Pydantic request and response models for the Phi-2 Serving API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Input for `/generate` and `/generate/stream`."""

    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=1024)


class GenerateResponse(BaseModel):
    """Synchronous generation result returned by `/generate`."""

    response: str
    prompt: str
    tokens_used: int


class ModelInfo(BaseModel):
    """Static info about the loaded model, returned by `/model/info`."""

    model_name: str
    adapter_path: str
    load_time_seconds: float


class HealthResponse(BaseModel):
    """Liveness payload returned by `/health`."""

    status: str
    model_loaded: bool
