"""Tests for the ingestion pipeline (no real PDF or ChromaDB needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import chunk_text, extract_text


def test_chunk_text_basic():
    pages = [(1, " ".join([f"word{i}" for i in range(600)]))]
    chunks = chunk_text(pages)
    assert len(chunks) > 1
    for c in chunks:
        word_count = len(c["text"].split())
        assert word_count <= 512 + 10  # slight tolerance


def test_chunk_text_overlap():
    pages = [(1, " ".join([f"word{i}" for i in range(700)]))]
    chunks = chunk_text(pages)
    # Verify overlap: last N words of chunk[0] == first N words of chunk[1]
    words0 = chunks[0]["text"].split()
    words1 = chunks[1]["text"].split()
    from src.config import CHUNK_OVERLAP
    assert words0[-CHUNK_OVERLAP:] == words1[:CHUNK_OVERLAP]


def test_chunk_text_single_page():
    pages = [(1, "short text")]
    chunks = chunk_text(pages)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1


def test_chunk_metadata():
    pages = [(3, " ".join(["word"] * 100))]
    chunks = chunk_text(pages)
    assert all(c["page"] == 3 for c in chunks)


@patch("src.ingestion._get_collection")
def test_ingest_pdf(mock_get_coll, tmp_path):
    """Full pipeline with mocked ChromaDB and PDF."""
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": []}
    mock_get_coll.return_value = mock_col

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("src.ingestion.extract_text", return_value=[(1, "This is page one content " * 20)]):
        from src.ingestion import ingest_pdf
        result = ingest_pdf(pdf_path, doc_id="test123")

    assert result["doc_id"] == "test123"
    assert result["chunks"] > 0
    mock_col.add.assert_called_once()


@patch("src.ingestion._get_collection")
def test_retrieve(mock_get_coll):
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
