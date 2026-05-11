# Athena AI OS with Ollama models

A local-first reasoning system built on Ollama with support for multiple LLMs, embeddings, and advanced retrieval capabilities.

## Features

- **Local LLM Integration**: Seamless integration with Ollama for running models locally
- **Multi-Model Support**: 
  - phi3 for general reasoning
  - deepseek-coder for code generation
  - nomic-embed-text for embeddings
  - llama3 for additional reasoning tasks
- **Async/Streaming**: Built on httpx for efficient async operations
- **Vector Embeddings**: 768-dimensional embeddings for semantic search
- **Batch Processing**: Parallel embedding generation with asyncio

## Project Structure

```
backend/
├── app/
│   ├── config.py          # Configuration and settings
│   ├── api/               # API routes
│   ├── core/
│   │   ├── llm/           # LLM client implementations
│   │   ├── chunking/      # Text chunking strategies
│   │   ├── extraction/    # Data extraction
│   │   ├── memory/        # Memory management
│   │   └── retrieval/     # Retrieval augmented generation
│   ├── models/            # Data models and schemas
│   ├── repositories/      # Data access layer
│   ├── services/          # Business logic
│   └── workers/           # Background tasks
├── test_client.py         # Phase 1 test harness
└── requirements.txt       # Python dependencies

frontend/
└── src/
    ├── components/        # React components
    └── hooks/             # Custom React hooks
```

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running on `localhost:11434`
- Models: phi3:latest, deepseek-coder:latest, nomic-embed-text:latest, llama3:latest

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# From project root
python backend/test_client.py
```

Expected test output:
```
[1/6] health_check ✓  models found: phi3:latest, deepseek-coder:latest, ...
[2/6] generate    ✓  chars=312  first50=The capital of France is Paris ...
[3/6] stream      ✓  tokens=47  first_token=The
[4/6] embed       ✓  dim=768    first3=[0.021, -0.003, 0.147]
[5/6] embed_batch ✓  2 vectors  elapsed=0.31s
[6/6] routing     ✓  code prompt → deepseek-coder:latest
```

## Configuration

Configuration is managed via `backend/app/config.py` using Pydantic settings.

Key settings:
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `LLM_MODEL`: Default reasoning model
- `CODE_MODEL`: Model for code generation tasks
- `EMBED_MODEL`: Model for text embeddings
- `EMBED_DIMENSIONS`: Embedding vector dimension (768)

## Architecture

### OllamaClient

The core LLM client providing:
- **generate()**: Single-shot LLM inference
- **stream()**: Token-by-token streaming responses
- **embed()**: Text to vector embedding
- **embed_batch()**: Batch embedding with parallel processing
- **health_check()**: Service availability verification
- **_route_model()**: Automatic model selection based on prompt type

### Key Design Decisions

1. **httpx over requests**: Async-first HTTP client for non-blocking I/O
2. **120s timeout**: Accommodates CPU inference (phi3: 8-15s, llama3: 45s)
3. **Smart model routing**: Automatically selects code model for code prompts
4. **Streaming support**: Real-time token delivery via `/api/generate`

## Development

### Adding New Models

Update `backend/app/config.py`:
```python
NEW_MODEL: str = "model-name:latest"
```

### Testing Custom Queries

Create test cases in `backend/test_client.py` following the existing pattern.

## API Endpoints

Currently in development. Phase 1 focuses on local client validation.

## License

MIT

## Author

Ujwal Doijode
