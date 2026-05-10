import httpx
import json
import asyncio
import time
from typing import AsyncGenerator
from app.core.llm.base import BaseLLMClient
from app.config import get_settings

settings = get_settings()

class OllamaClient(BaseLLMClient):

    """
    Local LLM client wrapping Ollama's REST API.
    
    Async Ollama client using httpx.AsyncClient.
 
    Key design decisions
    ────────────────────
    1. httpx over requests
       requests is synchronous; calling it inside an async function would
       block the entire event loop.  httpx.AsyncClient integrates natively
       with asyncio.
 
    2. 120-second timeout
       CPU inference for phi3 takes 8 to 15 s per response; llama3 up to 45 s.
       The default httpx timeout is 5 s — that would time out on every call.
       We set read=120 and keep connect/write lower to catch network issues fast.
 
    3. Model routing
       Rather than exposing raw model names to callers, generate() checks
       whether the prompt looks like a code task and picks the right model
       automatically.  This keeps all routing logic in one place.
 
    4. Streaming via /api/generate with stream=True
       Ollama returns newline-delimited JSON objects while generating.
       We parse each line and yield only the "response" field so callers
       receive clean text deltas.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.primary_model = settings.primary_model if hasattr(settings, 'primary_model') else settings.LLM_MODEL
        self.code_model = settings.code_model if hasattr(settings, 'code_model') else settings.CODE_MODEL
        self.embed_model = settings.embed_model if hasattr(settings, 'embed_model') else settings.EMBED_MODEL
        self._client = httpx.AsyncClient(timeout=120.0)

    def _route_model(self, prompt: str) -> str:
        """Route prompts to appropriate model based on content."""
        if "def " in prompt or "class " in prompt or "import " in prompt:
            return self.code_model
        return self.primary_model

    async def generate(self, prompt:str, system:str = "", system_prompt: str = "", use_code_model: bool = False) -> str:
        # Support both system and system_prompt parameters for compatibility
        sys_msg = system_prompt or system
        model = self.code_model if use_code_model else self._route_model(prompt)

        payload = {
            "model": model,
            "prompt": prompt,
            "system": sys_msg,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": getattr(settings, 'LLM_CONTEXT_WINDOW', 4096),
                "num_predict": getattr(settings, 'MAX_TOKENS_RESPONSE', 512),
            }
        }

        start = time.monotonic()
        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        elapsed = time.monotonic() - start

        result = response.json()
        print(f"[Athena] generate() | model={model} |"
              f"latency={elapsed:.2f}s | "
              f"tokens={result.get('eval_count', '?')}"
              )
        
        return result["response"]
    
    async def stream(self, prompt: str, system: str = "", use_code_model: bool = False) -> AsyncGenerator[str, None]:
        model = self.code_model if use_code_model else self._route_model(prompt)

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": getattr(settings, 'LLM_CONTEXT_WINDOW', 4096),
                "num_predict": getattr(settings, 'MAX_TOKENS_RESPONSE', 512),
            }
        }

        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
    
    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.embed_model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> dict:
        try:
            response = await self._client.get(
                f"{self.base_url}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return {
                "status": "ok",
                "models": model_names
            }
        except Exception as e:
            return {
                "status": "error",
                "models": [],
                "error": str(e)
            }
        
    async def aclose(self):
        await self._client.aclose()