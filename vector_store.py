"""
vector_store.py — Pinecone managed-service wrapper.

Handles index creation, upsertion, and similarity search.
"""

import hashlib
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_POOL_THREADS,
    TOP_K,
)


# ── Pinecone client (module-level singleton) ──────────────────────────────────
# pool_threads controls the size of the urllib3 connection pool used internally
# by the Pinecone SDK.  Async upsert batches run inside this pool, so set it
# to match or exceed the number of concurrent batch workers you intend to use.

_pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=PINECONE_POOL_THREADS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk_id(source: str, page: int, chunk_idx: int) -> str:
    """Deterministic, URL-safe ID for a chunk."""
    raw = f"{source}::p{page}::c{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── public API ────────────────────────────────────────────────────────────────

def get_or_create_index(
    index_name: str = PINECONE_INDEX_NAME,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """
    Return a handle to the Pinecone index, creating it if it doesn't exist.

    Args:
        index_name: Name of the Pinecone index.
        dimension:  Embedding vector length.
        metric:     Distance metric ('cosine', 'euclidean', or 'dotproduct').
        cloud:      Cloud provider for serverless spec.
        region:     Cloud region for serverless spec.

    Returns:
        Pinecone Index object.
    """
    existing = [idx.name for idx in _pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' …")
        _pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"✓ Index '{index_name}' created.")
    else:
        print(f"✓ Using existing index '{index_name}'.")

    return _pc.Index(index_name)


def upsert_chunks(
    index,
    chunks: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 100,
) -> None:
    """
    Upsert chunk vectors and metadata into Pinecone using parallel async batches.

    Each batch is dispatched concurrently via the SDK's internal thread pool
    (sized by ``PINECONE_POOL_THREADS``).  Futures are collected after all
    batches are submitted so the call blocks until every batch is confirmed.

    Args:
        index:      Pinecone Index object.
        chunks:     List of chunk dicts (text, source, page, chunk_idx).
        embeddings: Parallel list of embedding vectors.
        batch_size: Number of vectors per upsert call.
    """
    assert len(chunks) == len(embeddings), "chunks and embeddings must have the same length"

    vectors = [
        {
            "id": _chunk_id(c["source"], c["page"], c["chunk_idx"]),
            "values": emb,
            "metadata": {
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "chunk_idx": c["chunk_idx"],
            },
        }
        for c, emb in zip(chunks, embeddings)
    ]

    # Dispatch all batches asynchronously, then collect results.
    futures = []
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
        batch = vectors[i : i + batch_size]
        futures.append(index.upsert(vectors=batch, async_req=True))

    # Block until every future resolves (raises on any error).
    for future in futures:
        future.get()

    print(f"\n✓ Upserted {len(vectors)} vectors to Pinecone.")


def query_index(
    index,
    query_embedding: list[float],
    top_k: int = TOP_K,
    filter: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Args:
        index:           Pinecone Index object.
        query_embedding: Query vector.
        top_k:           Number of results to return.
        filter:          Optional Pinecone metadata filter dict.

    Returns:
        List of result dicts with keys: id, score, text, source, page.
    """
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )

    results = []
    for match in response.matches:
        results.append(
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", ""),
                "page": match.metadata.get("page", -1),
            }
        )
    return results


def delete_index(index_name: str = PINECONE_INDEX_NAME) -> None:
    """Permanently delete a Pinecone index. Use with caution."""
    _pc.delete_index(index_name)
    print(f"✓ Deleted index '{index_name}'.")
