"""Tests for the ingestion pipeline (no real ChromaDB needed)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion import chunk_text, extract_text


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    """Chunks must never exceed CHUNK_SIZE words."""
    pages = [(1, " ".join([f"word{i}" for i in range(600)]))]
    chunks = chunk_text(pages)
    assert len(chunks) > 1
    for c in chunks:
        word_count = len(c["text"].split())
        assert word_count <= CHUNK_SIZE


def test_chunk_text_overlap():
    """Last CHUNK_OVERLAP words of chunk[N] must equal first CHUNK_OVERLAP words of chunk[N+1]."""
    # Need enough words to produce at least 2 chunks.
    pages = [(1, " ".join([f"word{i}" for i in range(CHUNK_SIZE + CHUNK_OVERLAP + 10)]))]
    chunks = chunk_text(pages)
    assert len(chunks) >= 2, "Expected at least two chunks to verify overlap"

    words0 = chunks[0]["text"].split()
    words1 = chunks[1]["text"].split()

    # The first chunk is exactly CHUNK_SIZE words long.
    assert len(words0) == CHUNK_SIZE

    # The overlap: words0[-CHUNK_OVERLAP:] == words1[:CHUNK_OVERLAP]
    assert words0[-CHUNK_OVERLAP:] == words1[:CHUNK_OVERLAP], (
        f"Overlap mismatch: tail={words0[-CHUNK_OVERLAP:]}, head={words1[:CHUNK_OVERLAP]}"
    )


def test_chunk_text_single_page_short_text():
    """Text shorter than CHUNK_SIZE produces exactly one chunk."""
    pages = [(1, "short text")]
    chunks = chunk_text(pages)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"] == "short text"


def test_chunk_text_metadata_page():
    """All chunks from a page carry the correct page number."""
    pages = [(3, " ".join(["word"] * 100))]
    chunks = chunk_text(pages)
    assert all(c["page"] == 3 for c in chunks)


def test_chunk_text_chunk_index_sequential():
    """chunk_index values must be 0, 1, 2, ... across all pages."""
    pages = [
        (1, " ".join([f"a{i}" for i in range(600)])),
        (2, " ".join([f"b{i}" for i in range(600)])),
    ]
    chunks = chunk_text(pages)
    for expected, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == expected


def test_chunk_text_exact_chunk_size():
    """Exactly CHUNK_SIZE words produces a single chunk."""
    pages = [(1, " ".join([f"w{i}" for i in range(CHUNK_SIZE)]))]
    chunks = chunk_text(pages)
    assert len(chunks) == 1
    assert len(chunks[0]["text"].split()) == CHUNK_SIZE


# ── ingest_pdf ────────────────────────────────────────────────────────────────

@patch("src.ingestion._get_collection")
def test_ingest_pdf(mock_get_coll, tmp_path):
    """Full pipeline with mocked ChromaDB and mocked PDF extraction."""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": []}
    mock_get_coll.return_value = mock_col

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content here")

    with patch("src.ingestion.extract_text", return_value=[(1, "This is page one content " * 20)]):
        from src.ingestion import ingest_pdf
        result = ingest_pdf(pdf_path, doc_id="test123")

    assert result["doc_id"] == "test123"
    assert result["chunks"] > 0
    mock_col.add.assert_called_once()


@patch("src.ingestion._get_collection")
def test_ingest_pdf_replaces_existing_chunks(mock_get_coll, tmp_path):
    """Re-ingesting a doc deletes old chunks before adding new ones."""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": ["doc1_0", "doc1_1"]}
    mock_get_coll.return_value = mock_col

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 x")

    with patch("src.ingestion.extract_text", return_value=[(1, "hello " * 10)]):
        from src.ingestion import ingest_pdf
        ingest_pdf(pdf_path, doc_id="doc1")

    mock_col.delete.assert_called_once_with(ids=["doc1_0", "doc1_1"])
    mock_col.add.assert_called_once()


# ── retrieve ──────────────────────────────────────────────────────────────────

@patch("src.ingestion._get_collection")
def test_retrieve_normal(mock_get_coll):
    """Retrieve returns correctly shaped dicts with computed score."""
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [[{"doc_id": "doc1", "page": 2, "filename": "test.pdf"}]],
        "distances": [[0.15]],
    }
    mock_get_coll.return_value = mock_col

    from src.ingestion import retrieve
    results = retrieve("test query", doc_id="doc1", top_k=1)
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.85, abs=0.01)
    assert results[0]["page"] == 2
    assert results[0]["filename"] == "test.pdf"


@patch("src.ingestion._get_collection")
def test_retrieve_empty_results_no_crash(mock_get_coll):
    """retrieve() returns empty list when ChromaDB returns no documents."""
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_get_coll.return_value = mock_col

    from src.ingestion import retrieve
    results = retrieve("nothing here", top_k=5)
    assert results == []


@patch("src.ingestion._get_collection")
def test_retrieve_none_documents_no_crash(mock_get_coll):
    """retrieve() returns empty list when ChromaDB documents list is None/empty."""
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": None,
        "metadatas": None,
        "distances": None,
    }
    mock_get_coll.return_value = mock_col

    from src.ingestion import retrieve
    results = retrieve("nothing", top_k=5)
    assert results == []
