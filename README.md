# Phi-2 Serving API

A local REST API that serves Phi-2 behind FastAPI, with an optional MLX
LoRA adapter from the companion `ml-experiment` project. It runs natively
on Apple Silicon with MLX for fast on-device inference, and ships a
portable Docker image that falls back to PyTorch CPU. Both expose the same
endpoints and support synchronous and streaming generation.

When the MLX adapter is present, the native backend serves the fine-tuned
model; otherwise, and always in the container, it serves base Phi-2 (see
[Known Limitations](#known-limitations)).

## Why serve it this way?

A model is only useful once something can call it. Wrapping
Phi-2 in a small HTTP API turns a pile of weights into a service: the
model loads once at process start and stays resident, so every request
reuses it instead of paying the multi-second load cost again. MLX matters
here because it runs the model directly on the M4 GPU and Neural Engine
through Metal, which is dramatically faster than CPU inference. MLX is
macOS-only though, so the same API also has a PyTorch CPU backend that
runs anywhere in a container. That split is the point of the project: one
API surface, two runtimes chosen by where you deploy.

## Hardware & Stack

- M4 MacBook Pro (Apple Silicon, arm64)
- Python 3.11
- FastAPI + uvicorn
- MLX (`mlx-lm`) for native macOS inference
- PyTorch + Transformers for the portable Docker image
- Pydantic for request/response validation
- Base model: `microsoft/phi-2`

## What this demonstrates

- Loading a large model once and holding it as a process-wide singleton
  behind an async lifespan, so requests never re-pay load cost.
- Streaming token-by-token generation over HTTP with `StreamingResponse`.
- A single API contract backed by two swappable inference backends
  (MLX native, PyTorch containerized).
- Request logging middleware, typed request/response validation, and
  liveness/readiness semantics (`503` while the model is warming).

## Prerequisites

- Phi-2 weights cached locally via Hugging Face (`microsoft/phi-2`). The
  first model load fetches them if missing.
- Optional: a LoRA adapter trained in the companion `ml-experiment`
  project. The MLX backend applies it when present; if the adapter
  directory is missing, the base model loads instead. The adapter path
  defaults to `~/ml-experiment/experiments/2026-05-04-0014` and can be
  overridden with the `ADAPTER_PATH` environment variable.

  The adapter is stored in MLX format, so it is **only** applied by the
  native MLX backend. The PyTorch/Docker backend always serves base
  Phi-2.

## Installation (native macOS / MLX)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (native macOS / MLX)

```bash
./run.sh
```

This starts uvicorn on `0.0.0.0:8000` with `--reload`. The model is
loaded once at startup via FastAPI's lifespan event and held in memory
for the life of the process.

## Run with Docker (PyTorch CPU)

The Docker image runs the PyTorch CPU backend, which works on any
platform. MLX cannot run inside a Linux container because it needs the
macOS Metal framework, so containerized deployment uses PyTorch instead.

```bash
docker compose up --build
```

The image serves base Phi-2 on `http://localhost:8000`. Your Hugging Face
cache is mounted from the host so weights are not re-downloaded inside the
container.

To run detached:
```bash
docker compose up -d --build
```

To stop:
```bash
docker compose down
```

CPU inference is much slower than native MLX. The gap closes on a machine
with a CUDA GPU, which this CPU-only image does not use.

## Endpoints

| Method | Path                | Description                                       |
| ------ | ------------------- | ------------------------------------------------- |
| GET    | `/health`           | Liveness probe + model-loaded flag                |
| GET    | `/model/info`       | Model name, adapter path, load time in seconds    |
| POST   | `/generate`         | Synchronous completion for a prompt               |
| POST   | `/generate/stream`  | Streamed tokens, newline-separated                |

`503` is returned when the model isn't loaded; `422` for invalid input.

### Example curl commands

```bash
# Health
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Generate
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello, my name is", "max_tokens": 64}'

# Stream
curl -N -X POST http://localhost:8000/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello, my name is", "max_tokens": 64}'
```

## Performance

Throughput is measured by `scripts/benchmark.py`, which streams from
`/generate/stream`, counts the tokens the server emits, and reports the
median tokens/sec over several runs (the first run is discarded as
warmup). Counting streamed tokens works identically against both
backends, so the rows are directly comparable.

With the server running:

```bash
python scripts/benchmark.py --url http://localhost:8000 --label "MLX (native)"
```

Measured on an M4 MacBook Pro, prompt `"Hello, my name is"`, median of 3
runs (both backends emitted the same 15 tokens, so the rows are directly
comparable):

| Backend            | Runtime      | Throughput            | Notes            |
| ------------------ | ------------ | --------------------- | ---------------- |
| Native (macOS)     | MLX          | 14.3 tok/s (ttft 189ms) | GPU + Neural Engine via Metal |
| Docker             | PyTorch CPU  | 1.9 tok/s (ttft 695ms) | CPU only, base Phi-2 |

On-device MLX is about 7.5x faster than PyTorch CPU here. The gap would
close on a machine with a CUDA GPU, which this CPU-only image does not use.

`/health` and `/model/info` perform no inference and return in single-digit
milliseconds.

## Request Logging

Every request is logged to `logs/requests.log` via FastAPI middleware.
Each line records:

- timestamp
- endpoint path
- prompt, truncated to 100 characters (only for POSTs to `/generate*`)
- response time in milliseconds
- HTTP status code

The `logs/` directory is created on startup if it doesn't exist and is
git-ignored.

## Known Limitations

- The base Phi-2 model is not instruction-tuned and tends to hallucinate
  URLs and loop on open-ended prompts. The LoRA adapter helps for
  in-domain prompts only, and only on the MLX backend.
- The adapter is produced by the separate `ml-experiment` project and is
  not vendored here. If its directory is absent, the MLX backend loads
  base Phi-2 with no error, so serving the fine-tuned model requires the
  adapter to be present locally (or `ADAPTER_PATH` pointed at it).
- The Docker/PyTorch backend serves base Phi-2 without the adapter,
  because the adapter is in MLX format. `/model/info` still reports the
  configured `adapter_path` even when no adapter was applied.
- For a streaming response, the logged `response_time_ms` captures
  time-to-first-byte, not total stream duration, because the middleware
  records elapsed time when the response headers are sent.
- `tokens_used` in the response is approximated by whitespace-splitting
  the output, not by re-tokenizing. It is indicative, not exact.
- No authentication. Intended for local use; do not expose to the network
  without adding auth.

## Production Extensions

- Add API key authentication (e.g. a FastAPI dependency that checks an
  `Authorization` header).
- Add rate limiting (e.g. `slowapi`).
- Add HTTPS via an nginx reverse proxy in front of uvicorn.
- Swap Phi-2 for an instruction-tuned model (e.g. Phi-3-mini-instruct,
  Llama-3-Instruct) to remove the looping/hallucination issues.
- Serve on a CUDA GPU to close the gap with native MLX.

## Project Structure

```
phi2-serving/
├── app/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic request/response models
│   ├── model.py              # MLX model singleton + generate/stream
│   ├── main.py               # FastAPI app (MLX), endpoints, middleware
│   ├── model_torch.py        # PyTorch CPU backend for Docker
│   └── main_torch.py         # FastAPI app using the PyTorch backend
├── logs/
│   └── requests.log          # auto-generated request log (gitignored)
├── scripts/
│   └── benchmark.py          # throughput benchmark against a running server
├── tests/
│   └── test_api.py           # httpx-based smoke tests
├── Dockerfile                # PyTorch CPU image
├── docker-compose.yml
├── run.sh                    # native MLX uvicorn launcher
├── requirements.txt          # native macOS / MLX dependencies
├── requirements-linux.txt    # Docker / PyTorch dependencies
├── README.md
└── CLAUDE.md                 # architecture decisions & extension notes
```

## Acknowledgements

- Microsoft for the [Phi-2](https://huggingface.co/microsoft/phi-2) base
  model.
- The Apple MLX team for [`mlx-lm`](https://github.com/ml-explore/mlx-examples).
- FastAPI, uvicorn, and Pydantic for the serving stack.
- Built with assistance from Claude (Anthropic) for code generation and
  Claude Code for project scaffolding. All system design, implementation,
  and testing are my own.
