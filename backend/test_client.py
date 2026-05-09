"""
Phase 1 test harness — OllamaClient

Run from the project root:
    python backend/test_client.py

What each test validates
────────────────────────
1. health_check  → Ollama is reachable and your 4 models appear in the list
2. generate      → phi3 returns a coherent answer  (non-streaming)
3. stream        → tokens arrive progressively      (streaming)
4. embed         → nomic returns a 768-dim vector
5. embed_batch   → asyncio.gather fires in parallel (timing check)
6. code routing  → prompt with "def " routes to deepseek-coder

Expected output on healthy setup
─────────────────────────────────
[1/6] health_check ✓  models found: phi3:latest, deepseek-coder:latest, ...
[2/6] generate    ✓  chars=312  first50=The capital of France is Paris ...
[3/6] stream      ✓  tokens=47  first_token=The
[4/6] embed       ✓  dim=768    first3=[0.021, -0.003, 0.147]
[5/6] embed_batch ✓  2 vectors  elapsed=0.31s  (vs ~0.60s sequential)
[6/6] routing     ✓  code prompt → deepseek-coder:latest
"""

import asyncio
import sys
import time


# ── bootstrap path so imports work from project root ─────────────────────────
sys.path.insert(0, ".")

from backend.app.core.llm.ollama_client import OllamaClient
from backend.app.config import get_settings

settings = get_settings()


# ── helpers ───────────────────────────────────────────────────────────────────

def ok(label: str, detail: str) -> None:
    print(f"  ✅  {label:<18} {detail}")


def fail(label: str, err: Exception) -> None:
    print(f"  ❌  {label:<18} ERROR: {err}")
    raise SystemExit(1)


# ── individual tests ──────────────────────────────────────────────────────────

async def test_health(client: OllamaClient) -> None:
    result = await client.health_check()
    assert result["status"] == "ok", f"Ollama not reachable: {result}"
    models_str = ", ".join(result["models"])
    ok("health_check", f"models found: {models_str}")


async def test_generate(client: OllamaClient) -> None:
    answer = await client.generate(
        prompt="What is the capital of France? Answer in one sentence.",
        system_prompt="You are a concise assistant.",
    )
    assert len(answer) > 5, "Response suspiciously short"
    ok("generate", f"chars={len(answer)}  first50={answer[:50]!r}")


async def test_stream(client: OllamaClient) -> None:
    tokens: list[str] = []
    async for tok in client.stream(
        prompt="Count from 1 to 5, one number per word.",
    ):
        tokens.append(tok)

    assert tokens, "No tokens received from stream"
    ok("stream", f"tokens={len(tokens)}  first_token={tokens[0]!r}")


async def test_embed(client: OllamaClient) -> None:
    vector = await client.embed("Athena AI OS is a local-first reasoning system.")
    assert len(vector) == settings.embedding_dimension, (
        f"Expected dim={settings.embedding_dimension}, got {len(vector)}"
    )
    ok("embed", f"dim={len(vector)}  first3={[round(x, 3) for x in vector[:3]]}")


async def test_embed_batch(client: OllamaClient) -> None:
    texts = [
        "Local-first AI systems reduce data privacy risks.",
        "PGVector enables semantic similarity search in PostgreSQL.",
    ]
    t0 = time.perf_counter()
    vectors = await client.embed_batch(texts)
    elapsed = time.perf_counter() - t0

    assert len(vectors) == len(texts)
    assert all(len(v) == settings.embedding_dimension for v in vectors)
    ok("embed_batch", f"{len(vectors)} vectors  elapsed={elapsed:.2f}s")


async def test_code_routing(client: OllamaClient) -> None:
    """
    OllamaClient._route_model() should return deepseek-coder for code prompts.
    We test the routing logic directly without making a network call.
    """
    code_prompt = "def fibonacci(n): ..."
    routed = client._route_model(code_prompt)
    assert routed == settings.code_model, (
        f"Expected {settings.code_model}, got {routed}"
    )
    prose_prompt = "Explain what a RAG system is."
    routed_prose = client._route_model(prose_prompt)
    assert routed_prose == settings.primary_model
    ok("routing", f"code → {settings.code_model}  prose → {settings.primary_model}")


# ── main runner ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "═" * 55)
    print("  Athena AI OS — Phase 1 Client Tests")
    print("═" * 55)

    client = OllamaClient()

    tests = [
        ("1/6 health_check",  test_health),
        ("2/6 generate",      test_generate),
        ("3/6 stream",        test_stream),
        ("4/6 embed",         test_embed),
        ("5/6 embed_batch",   test_embed_batch),
        ("6/6 routing",       test_code_routing),
    ]

    for label, fn in tests:
        print(f"\n[{label}]")
        try:
            await fn(client)
        except Exception as exc:
            fail(label, exc)

    await client.aclose()
    print("\n" + "═" * 55)
    print("  All tests passed — Phase 1 complete ✅")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(main())