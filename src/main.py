"""
Paperwise FastAPI application.

Endpoints:
  POST /documents/upload  — ingest a PDF
  GET  /documents         — list all documents
  DELETE /documents/{id}  — remove a document
  POST /query             — ask a question
  POST /query/stream      — streaming answer
  GET  /health            — health check
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src import ingestion, llm
from src.config import UPLOAD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# PDF magic bytes: %PDF
_PDF_MAGIC = b"%PDF"
# 50 MB upload limit
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate that the Anthropic API key is configured at startup.
    # config.py raises ValueError during module import if key is missing;
    # we catch it here to surface a clear startup error rather than a cryptic
    # AttributeError later.
    try:
        from src.config import ANTHROPIC_API_KEY  # noqa: F401 — triggers validation
    except ValueError as exc:
        logger.critical("Startup aborted: %s", exc)
        raise RuntimeError(str(exc)) from exc

    logger.info("Paperwise starting up")
    yield
    logger.info("Paperwise shutting down")


app = FastAPI(
    title="Paperwise",
    description="Talk to your documents. Upload any PDF, ask anything, get cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    history: Optional[list[dict]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    usage: dict


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "paperwise"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), doc_id: Optional[str] = Form(None)):
    """Upload and ingest a PDF document."""
    # Extension check (fast path before reading content)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    # File size limit: 50 MB
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum upload size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # Magic bytes validation — must start with %PDF (25 50 44 46)
    if not content.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF (missing %PDF magic bytes).",
        )

    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    doc_id = doc_id or uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    dest.write_bytes(content)

    try:
        result = ingestion.ingest_pdf(dest, doc_id=doc_id)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {"success": True, **result}


@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    return {"documents": ingestion.list_documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the vector store."""
    removed = ingestion.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"success": True, "chunks_removed": removed}


@app.post("/query", response_model=QueryResponse)
def query_document(body: QueryRequest):
    """Ask a question. Returns the answer with cited page sources."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = llm.answer(body.query, doc_id=body.doc_id, history=body.history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/query/stream")
def query_stream(body: QueryRequest):
    """Streaming answer — returns text/event-stream of delta chunks."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def generator():
        try:
            for delta in llm.answer(body.query, doc_id=body.doc_id, history=body.history, stream=True):
                yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield "data: [ERROR]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
