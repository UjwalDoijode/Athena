from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (Distance, VectorParams, HnswConfigDiff,) 
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

_qdrant: AsyncQdrantClient = None

def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
    return _qdrant

async def init_qdrant():
    """
    Called once at startup.
    Creates the collection if it doesn't exist.
    HNSW index is built-in to Qdrant — no separate step needed.
    """

    client = get_qdrant_client()

    existing = await client.get_collection()
    existing_names = [c.name for c in existing.collections]

    if settings.QDRANT_COLLECTION not in existing_names:
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBED_DIMENSIONS,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            ),
        )
        logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION} created")
    else:
        logger.info(f"Qdrant collection already exists: {settings.QDRANT_COLLECTION}")

async def close_qdrant():
    global _qdrant
    if _qdrant:
        await _qdrant.close()
        _qdrant = None