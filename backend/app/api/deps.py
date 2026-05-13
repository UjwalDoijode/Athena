"""
creating dependency provider functions in FastAPI
"""
from fastapi import Request
from app.core.llm.ollama_client import OllamaClient
from qdrant_client import AsyncQdrantClient

def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama

def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant
