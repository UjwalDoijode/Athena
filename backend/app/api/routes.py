from fastapi import APIRouter
from app.api.endpoints import health, chat, embeddings, documents

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(embeddings.router, prefix='/embeddings', tags=['Embeddings'])
router.include_router(documents.router, prefix='/documents', tags=['Documents'])