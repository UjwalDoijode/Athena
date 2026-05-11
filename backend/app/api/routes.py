from fastapi import APIRouter
from app.api.endpoints import health, chat

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])