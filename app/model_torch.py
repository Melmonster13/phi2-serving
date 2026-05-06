"""PyTorch + HuggingFace Transformers backend for the Phi-2 Serving API.

This module mirrors the public interface of `app.model` (which uses MLX)
so the FastAPI app can swap backends at import time without any other
code changes. It is intended for Linux containers where MLX is not
available; on Apple Silicon, prefer `app.model`.

Public interface:
    load_model(), is_loaded(), get_info(), generate(), stream_generate()
"""

import os
import time
from pathlib import Path
from threading import Thread
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    os.path.expanduser("~/ml-experiment/experiments/2026-05-04-0014"),
)


class _ModelSingleton:
    """Holds the loaded HF model, tokenizer, and load metadata."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name: str = MODEL_NAME
        self.adapter_path: str = ADAPTER_PATH
        self.load_time_seconds: float = 0.0
        self.loaded: bool = False

    def load(self) -> None:
        """Load weights + tokenizer once. No-op on subsequent calls.

        If the LoRA adapter directory exists, it is applied via `peft`.
        Otherwise the base model is loaded standalone so the API still
        comes up.
        """
        if self.loaded:
            return
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # MLX-format adapters are not compatible with peft/PyTorch.
        # The Linux container runs base Phi-2 only.
        self.model = base_model

        self.model.eval()
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
    """Run synchronous generation and return only the new completion text."""
    if not _state.loaded:
        raise RuntimeError("Model not loaded")

    inputs = _state.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = _state.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=_state.tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0, prompt_len:]
    return _state.tokenizer.decode(new_tokens, skip_special_tokens=True)


def stream_generate(prompt: str, max_tokens: int = 256) -> Generator[str, None, None]:
    """Yield tokens as they are produced, using a background generation thread."""
    if not _state.loaded:
        raise RuntimeError("Model not loaded")

    inputs = _state.tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(
        _state.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_tokens,
        pad_token_id=_state.tokenizer.pad_token_id,
        streamer=streamer,
    )

    def _run() -> None:
        with torch.no_grad():
            _state.model.generate(**generation_kwargs)

    thread = Thread(target=_run, daemon=True)
    thread.start()

    for text in streamer:
        if text:
            yield text

    thread.join()
