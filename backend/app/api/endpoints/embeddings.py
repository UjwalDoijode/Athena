from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.deps import get_ollama
from app.core.llm.ollama_client import OllamaClient
from app.services.embedding_service import EmbeddingService
from typing import Literal


router = APIRouter()

class EmbedRequest(BaseModel):
    text: str

class ChunkEmbedRequest(BaseModel):
    text: str
    strategy: Literal["fixed", "sentence", "semantic"] = "sentence"
    chunk_size: int = 512
    overlap: int = 64

@router.post("/embed")
async def embed(
    request: EmbedRequest,
    ollama: OllamaClient = Depends(get_ollama),
):
    svc = EmbeddingService(ollama)
    results = await svc.chunk_and_embed(
        text=request.text,
        strategy=request.strategy,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return {
        "strategy": request.strategy,
        "chunk_count": len(results),
        "chunks": [
            {
                "index": r["chunk"].index,
                "text": r["text"][:120] + "..." if len(r["text"]) > 120 else r["text"],
                "embedding_preview": r["embedding"][:3]
            }
            for r in results
        ],
    }
