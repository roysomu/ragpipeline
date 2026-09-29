"""
embedder.py — Generate Google text-embedding-004 embeddings for text chunks.

Uses the same google-genai SDK already present for the Gemini LLM — no extra
dependency needed.
"""

from google import genai
from google.genai import types

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, GOOGLE_API_KEY


_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Return Google text-embedding-004 embeddings for a list of strings.

    Each text is embedded with task_type=RETRIEVAL_DOCUMENT for optimal
    retrieval quality.

    Args:
        texts:      List of strings to embed.
        batch_size: How many texts to send per API call.

    Returns:
        List of embedding vectors (list[float]), preserving input order.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )
        all_embeddings.extend([e.values for e in response.embeddings])
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Return an embedding vector for a single query string.

    Uses task_type=RETRIEVAL_QUERY so the vector is optimised for similarity
    search against document embeddings.

    Args:
        query: The user question.

    Returns:
        Embedding vector.
    """
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return response.embeddings[0].values
