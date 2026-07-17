#!/usr/bin/env python3
"""Measure generation throughput against a running Phi-2 Serving API.

Streams from `/generate/stream`, counts the tokens the server emits, and
reports tokens/sec plus time-to-first-token. Counting streamed tokens
gives a real generated-token count that works the same way against either
backend (MLX or PyTorch CPU), so the two rows in the README are directly
comparable.

Start the server first (`./run.sh` for MLX, or `docker compose up` for
PyTorch CPU), then run this against it:

    python scripts/benchmark.py --url http://localhost:8000

The first (warmup) run is discarded; the reported numbers are the median
of the timed runs. A markdown table row is printed at the end, ready to
paste into the README Performance section.
"""

import argparse
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class RunResult:
    tokens: int
    total_s: float
    ttft_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.total_s if self.total_s > 0 else 0.0


def one_run(url: str, prompt: str, max_tokens: int, timeout: float) -> RunResult:
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    tokens = 0
    ttft_s = 0.0
    start = time.perf_counter()
    with httpx.stream(
        "POST", f"{url}/generate/stream", json=payload, timeout=timeout
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if tokens == 0:
                ttft_s = time.perf_counter() - start
            tokens += 1
    total_s = time.perf_counter() - start
    return RunResult(tokens=tokens, total_s=total_s, ttft_s=ttft_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3, help="timed runs (median reported)")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", default="", help="row label for the printed table")
    args = parser.parse_args()

    print(f"Warming up against {args.url} ...")
    one_run(args.url, args.prompt, args.max_tokens, args.timeout)

    results = []
    for i in range(args.runs):
        res = one_run(args.url, args.prompt, args.max_tokens, args.timeout)
        results.append(res)
        print(
            f"  run {i + 1}: {res.tokens} tokens in {res.total_s:.2f}s "
            f"({res.tokens_per_s:.1f} tok/s, ttft {res.ttft_s * 1000:.0f}ms)"
        )

    tps = statistics.median(r.tokens_per_s for r in results)
    ttft_ms = statistics.median(r.ttft_s for r in results) * 1000
    tokens = statistics.median(r.tokens for r in results)

    print(f"\nMedian: {tps:.1f} tok/s | {tokens:.0f} tokens | ttft {ttft_ms:.0f}ms")
    label = args.label or "REPLACE_ME"
    print("\nMarkdown row:")
    print(f"| {label} | {tps:.1f} tok/s (ttft {ttft_ms:.0f}ms) |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
