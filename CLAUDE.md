# Architecture Decisions

## Singleton model loader
- `app/model.py` defines a module-level `_ModelSingleton` instance
  (`_state`) that owns the MLX model, tokenizer, and load metadata.
  Loading happens exactly once per process; `load_model()` is
  idempotent.
- The singleton is plain module state, not a global object passed
  through dependency injection. This is deliberate: the model is a
  process-wide resource and there is exactly one of it.

## Lifespan over `@app.on_event`
- The model is loaded inside FastAPI's `lifespan` async context manager
  (`app/main.py`). This keeps imports cheap (e.g. for tests that don't
  need the model) and ties load lifecycle to the app lifecycle.
- The `lifespan` shutdown branch is a no-op — MLX handles its own
  cleanup and there's no client to close.

## Middleware request logging
- A single HTTP middleware in `app/main.py` logs every request to
  `logs/requests.log` with timestamp, endpoint, prompt preview
  (truncated to 100 chars), response time in ms, and status code.
- For POSTs to `/generate*`, the body is read once to extract a prompt
  preview, then re-attached via `request._body = body` so the route
  handler still receives it. This is a small Starlette-internal hack
  but is the cleanest way to preview a body in middleware.

## Validation
- All request/response shapes live in `app/schemas.py`. `max_tokens` is
  bounded `[1, 1024]` and `prompt` must be non-empty; FastAPI returns
  422 automatically on violation.
- 503 is raised explicitly via `_require_loaded()` when the model isn't
  ready. This lets probes distinguish "process up, model warming" from
  fully healthy.

# Known Issues

- **Stream `response_time_ms` is misleading.** The middleware records
  elapsed time when `call_next(request)` returns, which for a
  `StreamingResponse` is when headers are sent, not when the stream
  ends. So `/generate/stream` lines in `requests.log` capture
  time-to-first-byte, not total stream duration.
- **`tokens_used` is approximate.** It's `len(text.split())`, not a
  true token count. Switching to `len(tokenizer.encode(text))` would
  fix this at a small CPU cost.
- **Adapter resolution is silent.** If the LoRA adapter directory is
  missing, the base model loads with no warning surfaced to the API
  client. `/model/info` still reports the configured `adapter_path`
  even when it wasn't actually applied.

# How to Extend

## Add an endpoint
1. Add a Pydantic model to `app/schemas.py` if needed.
2. Add a route in `app/main.py`. Call `_require_loaded()` first if it
   needs the model.
3. If it should appear in `requests.log` with a prompt preview, name the
   path so it starts with `/generate` — or generalize the middleware's
   path filter.

## Swap the model
- Edit `MODEL_NAME` and/or `ADAPTER_PATH` at the top of `app/model.py`.
  Both are module-level constants. The singleton picks them up on next
  process start.
- For an instruction-tuned model with a chat template, you'll likely
  want to call `tokenizer.apply_chat_template(...)` before passing the
  prompt to `mlx_lm.generate`. That logic belongs in `app/model.py`.

## Add authentication
- Add a FastAPI dependency that reads an `X-API-Key` header (or
  `Authorization: Bearer ...`) and raises 401 if invalid.
- Apply it at the app level via `dependencies=[Depends(...)]` on
  `FastAPI(...)` so every endpoint is protected, or per-route for finer
  control.
- Health probes typically stay unauthenticated; gate `/generate*` and
  `/model/info`.

## Adapter path configuration
- Currently hard-coded as `~/ml-experiment/experiments/2026-05-04-0014`
  in `app/model.py`. To make it configurable, read it from an
  environment variable (e.g. `PHI2_ADAPTER_PATH`) with the current
  value as the default. Same for `MODEL_NAME` if you want to switch
  base models without code changes.
