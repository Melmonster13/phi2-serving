"""MLX-backed Phi-2 model loader and inference helpers.

A module-level singleton (`_state`) holds the model and tokenizer so they
are loaded exactly once per process. The FastAPI app triggers loading
from its lifespan event; tests can import this module without paying
that cost until `load_model()` is called.
"""

import os
import time
from pathlib import Path
from typing import Generator, Optional

from mlx_lm import load, generate as mlx_generate
from mlx_lm import stream_generate as mlx_stream_generate


MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    os.path.expanduser("~/ml-experiment/experiments/2026-05-04-0014"),
)


class _ModelSingleton:
    """Holds the loaded MLX model, tokenizer, and load metadata."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name: str = MODEL_NAME
        self.adapter_path: str = ADAPTER_PATH
        self.load_time_seconds: float = 0.0
        self.loaded: bool = False

    def load(self) -> None:
        """Load weights + tokenizer once. No-op on subsequent calls.

        If the LoRA adapter directory is missing, the base model loads
        without an adapter so the API still comes up.
        """
        if self.loaded:
            return
        start = time.perf_counter()
        adapter = self.adapter_path if Path(self.adapter_path).exists() else None
        self.model, self.tokenizer = load(self.model_name, adapter_path=adapter)
        self.load_time_seconds = time.perf_counter() - start
        self.loaded = True


_state = _ModelSingleton()


def load_model() -> None:
    """Load the model into the module-level singleton (idempotent)."""
    _state.load()


def is_loaded() -> bool:
    """Return whether the model has been loaded into memory."""
    return _state.loaded


def get_info() -> dict:
    """Return model name, adapter path, and load time in seconds."""
    return {
        "model_name": _state.model_name,
        "adapter_path": _state.adapter_path,
        "load_time_seconds": _state.load_time_seconds,
    }


def generate(prompt: str, max_tokens: int = 256) -> str:
    """Run synchronous generation and return the full completion."""
    if not _state.loaded:
        raise RuntimeError("Model not loaded")
    return mlx_generate(
        _state.model,
        _state.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )


def stream_generate(prompt: str, max_tokens: int = 256) -> Generator[str, None, None]:
    """Yield tokens as they are produced by the model."""
    if not _state.loaded:
        raise RuntimeError("Model not loaded")
    for chunk in mlx_stream_generate(
        _state.model,
        _state.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        text = getattr(chunk, "text", None)
        if text is None:
            text = str(chunk)
        yield text
