from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.api.deps import get_ollama, get_qdrant
from pydantic import BaseModel
from typing import optional
from app.core.llm.ollama_client import OllamaClient
import json

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    system: str = ""
    stream: bool = False
    use_code_model: bool = False

async def chat(
        request: ChatRequest,
        ollama: OllamaClient = Depends(get_ollama),
):
    if request.stream:
        async def token_stream():
            async for token in ollama.stream(
                prompt=request.message,
                system=request.system,
                use_code_model=request.use_code_model,
            ):
              yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    response_text = await ollama.generate(
        prompt=request.message,
        system=request.system,
        use_code_model=request.use_code_model,
    )
    model_used = (
        ollama.code_model if request.use_code_model
        else ollama._route_model(request.message)
    )
    return {"response": response_text, "model": model_used}
