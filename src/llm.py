"""
LLM layer — Claude-powered Q&A with retrieved context.

Supports both single-turn and multi-turn (conversation history) modes.
"""

from __future__ import annotations

import logging
from typing import Iterator

import anthropic

from src.config import ANTHROPIC_API_KEY, CHAT_MODEL, TOP_K
from src.ingestion import retrieve

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """\
You are Paperwise, an AI assistant that answers questions about uploaded documents.

Rules:
1. Base your answers ONLY on the provided document excerpts.
2. Always cite the page number(s) your answer comes from, e.g. "(page 3)".
3. If the answer cannot be found in the excerpts, say so clearly — do not hallucinate.
4. Keep answers concise but complete. Use bullet points for lists.
5. If asked something outside the document scope, politely redirect.
"""


def build_context_block(chunks: list[dict]) -> str:
    lines = ["## Relevant document excerpts\n"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"### Excerpt {i} — {c['filename']} (page {c['page']}, relevance {c['score']})")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def answer(
    query: str,
    doc_id: str | None = None,
    history: list[dict] | None = None,
    stream: bool = False,
) -> dict:
    """
    Retrieve relevant chunks and ask Claude to answer the query.

    Parameters
    ----------
    query   : The user's question.
    doc_id  : Scope retrieval to a specific document (optional).
    history : Prior conversation turns [{"role": "user"|"assistant", "content": str}].
    stream  : If True, returns a generator of text deltas instead of the full response.

    Returns
    -------
    dict with keys: answer, sources, model, usage
    """
    chunks = retrieve(query, doc_id=doc_id, top_k=TOP_K)
    context = build_context_block(chunks)

    messages = list(history or [])
    messages.append({
        "role": "user",
        "content": f"{context}\n\n---\n\nQuestion: {query}",
    })

    client = _get_client()

    if stream:
        return _stream_answer(client, messages, chunks)

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    answer_text = response.content[0].text
    logger.info("Answered query (tokens: %d in / %d out)", response.usage.input_tokens, response.usage.output_tokens)

    return {
        "answer": answer_text,
        "sources": [{"filename": c["filename"], "page": c["page"], "score": c["score"]} for c in chunks],
        "model": CHAT_MODEL,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }


def _stream_answer(client: anthropic.Anthropic, messages: list[dict], chunks: list[dict]):
    """Yield text deltas from a streaming Claude response."""
    with client.messages.stream(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
# multi-turn history passed as messages array
