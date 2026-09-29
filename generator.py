"""
generator.py — Answer generation using Google Gemini (google-genai SDK v2).
"""

from google import genai
from google.genai import types

from config import GEMINI_MODEL, GOOGLE_API_KEY

_client = genai.Client(api_key=GOOGLE_API_KEY)


# ── prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the answer is not in the context, say "
    "\"I don't have enough information in the provided documents to answer that.\"\n\n"
    "Always cite the source document and page number when you use information "
    "from the context."
)


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Construct the full prompt from retrieved chunks.

    Args:
        query:          The user's question.
        context_chunks: List of dicts with 'text', 'source', 'page' keys.

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} | Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )
    context_text = "\n\n".join(context_parts)

    return (
        f"--- CONTEXT ---\n{context_text}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        f"--- ANSWER ---"
    )


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer for *query* grounded in *context_chunks*.

    Args:
        query:          The user's question.
        context_chunks: Retrieved chunks from Pinecone.

    Returns:
        Generated answer string.
    """
    prompt = build_prompt(query, context_chunks)
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )
    return response.text
