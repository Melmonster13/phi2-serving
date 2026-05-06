# Phi-2 Serving API

A local REST API that serves a fine-tuned Phi-2 model behind FastAPI and
MLX. Supports both synchronous and streaming generation.

## Hardware & Stack

- M4 MacBook Pro (Apple Silicon, arm64)
- Python 3.11
- FastAPI + uvicorn
- MLX (`mlx-lm`) for model loading and inference
- Pydantic for request/response validation
- Base model: `microsoft/phi-2`

## Prerequisites

- Phi-2 weights cached locally via Hugging Face (`microsoft/phi-2`).
  The first `mlx_lm.load(...)` call will fetch them if missing.
- LoRA adapter trained in the companion `ml-experiment` project, located
  at `~/ml-experiment/experiments/2026-05-04-0014`. If the path is
  missing, the API falls back to the base model.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

This starts uvicorn on `0.0.0.0:8000` with `--reload`. The model is
loaded once at startup via FastAPI's lifespan event and held in memory
for the life of the process.

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

## Performance Notes

- **Synchronous `/generate`**: ~7–9 tokens/sec on M4. Total latency
  scales linearly with `max_tokens`.
- **Streaming `/generate/stream`**: returns immediately; tokens flow
  asynchronously as the model produces them. Time-to-first-token is a
  better latency metric here than total elapsed time.
- **`/health` and `/model/info`**: sub-2ms — no inference is performed.

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
  in-domain prompts only.
- No authentication. Intended for local use; do not expose to the
  network without adding auth.
- `tokens_used` in the response is approximated by whitespace-splitting
  the output, not by re-tokenizing. It is indicative, not exact.

## Production Extensions

- Add API key authentication (e.g. a FastAPI dependency that checks an
  `Authorization` header).
- Add rate limiting (e.g. `slowapi`).
- Add HTTPS via an nginx reverse proxy in front of uvicorn.
- Swap Phi-2 for an instruction-tuned model (e.g. Phi-3-mini-instruct,
  Llama-3-Instruct) to remove the looping/hallucination issues.
- Deploy with Docker; pin Python and MLX versions in the image.

## Project Structure

```
phi2-serving/
├── app/
│   ├── __init__.py
│   ├── schemas.py       # Pydantic request/response models
│   ├── model.py         # MLX model singleton + generate/stream
│   └── main.py          # FastAPI app, endpoints, middleware logging
├── logs/
│   └── requests.log     # auto-generated request log (gitignored)
├── tests/
│   └── test_api.py      # httpx-based smoke tests
├── run.sh               # uvicorn launcher
├── requirements.txt
├── README.md
└── CLAUDE.md            # architecture decisions & extension notes
```

## Acknowledgements

- Microsoft for the [Phi-2](https://huggingface.co/microsoft/phi-2) base
  model.
- The Apple MLX team for [`mlx-lm`](https://github.com/ml-explore/mlx-examples).
- FastAPI, uvicorn, and Pydantic for the serving stack.
- Built with assistance from Claude (Anthropic) for code generation and 
  Claude Code for project scaffolding. All system design, implementation, 
  and testing are my own.
