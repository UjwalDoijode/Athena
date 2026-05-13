from fastapi import APIRouter, Depends
from app.api.deps import get_ollama, get_qdrant
from app.core.llm.ollama_client import OllamaClient
from qdrant_client import AsyncQdrantClient
from app.config import get_settings

settings = get_settings()
router = APIRouter()

@router.get("")
async def health_check(
    ollama: OllamaClient = Depends(get_ollama),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    ollama_health = await ollama.health_check()
    qdrant_info = await qdrant.get_collections(settings.QDRANT_COLLECTION)

    return {
        "status": "ok",
        "ollama": ollama_health,
        "qdrant": {
            "status": "ok",
            "collection": settings.QDRANT_COLLECTION,
            "vectors_count": qdrant_info.vectors_count,
        }
    }
