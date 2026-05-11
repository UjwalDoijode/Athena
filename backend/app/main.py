from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings
from app.api.routes import router
from app.core.llm.ollama_client import OllamaClient
from app.core.qdrant_client import init_qdrant, get_qdrant_client, close_qdrant
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama = OllamaClient()
    health = await ollama.health_check()
    
    if health.get("status") == "ok":
        logger.info(f" Athena AI OS starting — model: {settings.LLM_MODEL}")
    else:
        logger.warning(f" Ollama unreachable: {health.get('error')}")

    app.state.ollama = ollama

    #Qdrant
    await init_qdrant()
    app.state.qdrant = get_qdrant_client()
    logger.info("Qdrant ready!!!")

    yield

    await ollama.aclose()
    await close_qdrant()
    logger.info("Athena shut down cleanly")


app = FastAPI(
    title="Athena AI OS",
    description="Local-first AI Reasoning System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "athena-ai-os"}