# paperwise

> Talk to your documents. Upload any PDF, ask anything, get answers with exact page citations — powered by a RAG pipeline on Claude + ChromaDB.

## How it works

```
PDF → text extraction (pypdf) → chunking (512 tokens, 64-token overlap)
   → ChromaDB (cosine similarity, HNSW index)
   → Claude 3.5 Sonnet (retrieval-augmented generation)
   → answer + page citations
```

## Quick Start

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY

pip install -r requirements.txt
uvicorn src.main:app --reload
```

## API

### Upload a PDF
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@research.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "doc_id": "abc123"}'
```

Response:
```json
{
  "answer": "The main findings are... (page 4)",
  "sources": [
    { "filename": "research.pdf", "page": 4, "score": 0.91 },
    { "filename": "research.pdf", "page": 7, "score": 0.87 }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "usage": { "input_tokens": 1240, "output_tokens": 180 }
}
```

### Streaming answer
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise section 3"}'
```

### Multi-turn conversation
```json
{
  "query": "Can you elaborate on that?",
  "doc_id": "abc123",
  "history": [
    { "role": "user", "content": "What are the limitations?" },
    { "role": "assistant", "content": "The main limitations are... (page 12)" }
  ]
}
```

## Tech Stack

Python · FastAPI · ChromaDB · pypdf · Claude 3.5 Sonnet (Anthropic) · SSE streaming
