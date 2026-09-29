"""
rag_pipeline.py — Programmatic RAG pipeline interface.

Files arrive as raw bytes from Supabase; disk-based ingestion is handled
by the caller (app.py) before this pipeline is invoked.

Example:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()

    # Ingest a PDF already read from Supabase
    pipeline.ingest_bytes(pdf_bytes, "report.pdf")

    # Ask questions
    answer = pipeline.query("What is the main conclusion?")
    print(answer)
"""

from document_loader import load_pdf_from_bytes
from embedder import embed_query, embed_texts
from generator import generate_answer
from vector_store import get_or_create_index, query_index, upsert_chunks


class RAGPipeline:
    """
    High-level RAG pipeline combining ingestion and querying.

    Args:
        index_name: Pinecone index name (defaults to config value).
        top_k:      Number of retrieved context chunks per query.
        verbose:    Print progress messages when True.
    """

    def __init__(
        self,
        index_name: str | None = None,
        top_k: int | None = None,
        verbose: bool = True,
    ) -> None:
        from config import TOP_K

        self._top_k = top_k or TOP_K
        self._verbose = verbose
        kwargs = {"index_name": index_name} if index_name else {}
        self._index = get_or_create_index(**kwargs)

    # ── ingestion ──────────────────────────────────────────────────────────────

    def _embed_and_upsert(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        if self._verbose:
            print(f"Embedding {len(chunks)} chunks …")
        embeddings = embed_texts([c["text"] for c in chunks])
        upsert_chunks(self._index, chunks, embeddings)

    def ingest_bytes(self, data: bytes, filename: str) -> "RAGPipeline":
        """
        Ingest a PDF from raw bytes (as received from Supabase).

        Args:
            data:     Raw PDF bytes.
            filename: The document's original filename (used as source label).

        Returns:
            self (for chaining).
        """
        chunks = list(load_pdf_from_bytes(data, filename))
        if self._verbose:
            print(f"Loaded {len(chunks)} chunks from {filename}")
        self._embed_and_upsert(chunks)
        return self

    # ── querying ───────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        source_filter: str | None = None,
        return_context: bool = False,
    ) -> str | tuple[str, list[dict]]:
        """
        Query the RAG pipeline.

        Args:
            question:       The user's question.
            source_filter:  Optional: restrict retrieval to one PDF filename.
            return_context: If True, also return the retrieved context chunks.

        Returns:
            Generated answer string, or (answer, context_chunks) if
            return_context=True.
        """
        query_emb = embed_query(question)
        pinecone_filter = (
            {"source": {"$eq": source_filter}} if source_filter else None
        )
        context = query_index(
            self._index, query_emb, top_k=self._top_k, filter=pinecone_filter
        )

        if not context:
            answer = "No relevant context found. Have you indexed any documents?"
            return (answer, []) if return_context else answer

        answer = generate_answer(question, context)
        return (answer, context) if return_context else answer
