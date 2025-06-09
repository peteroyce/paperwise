"""
PDF ingestion pipeline.

1. Extract raw text from PDF pages (pypdf)
2. Split into overlapping chunks
3. Store chunks + embeddings in ChromaDB
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator

import chromadb
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

_chroma_client: chromadb.PersistentClient | None = None


def _get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _chroma_client.get_or_create_collection(
        name="paperwise",
        metadata={"hnsw:space": "cosine"},
    )


def extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)
    return pages


def chunk_text(pages: list[tuple[int, str]]) -> list[dict]:
    """Split page text into overlapping chunks with metadata.

    Each new chunk starts CHUNK_SIZE - CHUNK_OVERLAP words after the previous
    chunk's start, so the last CHUNK_OVERLAP words of chunk N are the first
    CHUNK_OVERLAP words of chunk N+1.
    """
    chunks = []
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_words = words[start:end]
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_num,
                "chunk_index": len(chunks),
            })
            if end == len(words):
                break
            # Advance by CHUNK_SIZE - CHUNK_OVERLAP so adjacent chunks share
            # CHUNK_OVERLAP words at their boundary.
            start += CHUNK_SIZE - CHUNK_OVERLAP
    logger.debug("Produced %d chunks", len(chunks))
    return chunks


def ingest_pdf(pdf_path: Path, doc_id: str | None = None) -> dict:
    """
    Full ingestion pipeline: extract → chunk → embed → store.

    Returns a summary dict with doc_id and chunk count.
    ChromaDB uses its built-in embedding function (all-MiniLM-L6-v2 by default).

    Cleanup rollback: if ChromaDB add raises after old chunks were deleted, the
    deletion is already committed (ChromaDB has no transactions), so we log the
    inconsistency. If the add itself fails the collection is left empty for this
    doc_id — on re-ingest the delete step is a no-op and ingestion retries cleanly.
    """
    doc_id = doc_id or hashlib.md5(pdf_path.read_bytes()).hexdigest()
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)

    collection = _get_collection()

    # Remove existing chunks for this doc (re-ingest support)
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Replaced %d existing chunks for doc %s", len(existing["ids"]), doc_id)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"doc_id": doc_id, "page": c["page"], "filename": pdf_path.name} for c in chunks]

    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as exc:
        logger.error(
            "ChromaDB add failed for doc_id=%s (%d chunks). "
            "Collection may be empty for this doc. Re-ingest to recover. Error: %s",
            doc_id, len(chunks), exc,
        )
        raise

    logger.info("Ingested %d chunks for doc_id=%s", len(chunks), doc_id)

    return {"doc_id": doc_id, "filename": pdf_path.name, "pages": len(pages), "chunks": len(chunks)}


def retrieve(query: str, doc_id: str | None = None, top_k: int = 5) -> list[dict]:
    """Semantic search over stored chunks. Optionally scoped to a single doc."""
    collection = _get_collection()
    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Bounds check: ChromaDB returns nested lists; guard against empty results.
    docs_list = results.get("documents") or []
    if not docs_list or not docs_list[0]:
        logger.debug("retrieve: no results found for query=%r doc_id=%r", query, doc_id)
        return []

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "page": meta.get("page"),
            "filename": meta.get("filename"),
            "doc_id": meta.get("doc_id"),
            "score": round(1 - dist, 4),
        })
    return chunks


def list_documents() -> list[dict]:
    """Return unique documents stored in the collection."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    seen = {}
    for meta in result["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": meta.get("filename")}
    return list(seen.values())


def delete_document(doc_id: str) -> int:
    """Delete all chunks for a document. Returns number of chunks removed."""
    collection = _get_collection()
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    return len(existing["ids"])
