from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Literal, Optional
from qdrant_client import AsyncQdrantClient
from app.api.deps import get_qdrant, get_ollama
from app.core.llm.ollama_client import OllamaClient
from app.services.document_service import DocumentService

router = APIRouter()

class IngestRequest(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    doc_type: str = "text"
    strategy: Literal["fixed", "sentence", "semantic"] = "sentence"
    metadata: dict = {}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None

@router.post("/ingest")
async def ingest_document(
        request: IngestRequest,
        qdrant: AsyncQdrantClient = Depends(get_qdrant),
        ollama: OllamaClient = Depends(get_ollama),
):
    svc = DocumentService(qdrant, ollama)
    return await svc.ingest(
        title=request.title,
        content=request.content,
        source=request.source,
        doc_type=request.doc_type,
        strategy=request.strategy,
        metadata=request.metadata,
    )

@router.post("/search")
async def search(
    request: SearchRequest,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    ollama: OllamaClient = Depends(get_ollama),
):
    svc = DocumentService(qdrant, ollama)
    results = await svc.search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    return {
        "query": request.query,
        "results": results,
    }

@router.get("/list")
async def list_documents(
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    from app.repositories.vector_repository import VectorRepository
    repo = VectorRepository(qdrant)
    return await repo.list_documents()

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    from app.repositories.vector_repository import VectorRepository
    repo = VectorRepository(qdrant)
    await repo.delete_document(document_id)
    return {"deleted": document_id}

@router.get("/info")
async def collection_info(qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    from app.repositories.vector_repository import VectorRepository
    repo = VectorRepository(qdrant)
    return await repo.collection_info()