import sys
import httpx


BASE_URL = "http://127.0.0.1:8000"


def _result(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def test_health() -> bool:
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    return _result("GET /health", ok, f"status={r.status_code}")


def test_model_info() -> bool:
    r = httpx.get(f"{BASE_URL}/model/info", timeout=30)
    data = r.json() if r.status_code == 200 else {}
    expected = {"model_name", "adapter_path", "load_time_seconds"}
    ok = r.status_code == 200 and expected.issubset(data.keys())
    return _result("GET /model/info", ok, f"keys={list(data.keys())}")


def test_generate() -> bool:
    payload = {"prompt": "Hello, my name is", "max_tokens": 16}
    r = httpx.post(f"{BASE_URL}/generate", json=payload, timeout=120)
    data = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and bool(data.get("response"))
    return _result("POST /generate", ok, f"status={r.status_code}")


def test_generate_stream() -> bool:
    payload = {"prompt": "Hello, my name is", "max_tokens": 16}
    chunks = []
    with httpx.stream(
        "POST", f"{BASE_URL}/generate/stream", json=payload, timeout=120
    ) as r:
        for line in r.iter_lines():
            if line:
                chunks.append(line)
    ok = len(chunks) > 0
    return _result("POST /generate/stream", ok, f"chunks={len(chunks)}")


def main() -> int:
    tests = [test_health, test_model_info, test_generate, test_generate_stream]
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            results.append(_result(t.__name__, False, f"exception: {e}"))
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
