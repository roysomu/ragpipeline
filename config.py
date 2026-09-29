"""
config.py — Centralised configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Read an env var or raise a clear error."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: '{key}'. "
            "Copy .env.example → .env and fill in your keys."
        )
    return val


# ── API keys ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = _require("PINECONE_API_KEY")
GOOGLE_API_KEY: str = _require("GOOGLE_API_KEY")
# OPENAI_API_KEY is no longer required — embeddings now use Google text-embedding-004.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Supabase (document storage) ───────────────────────────────────────────────
DATABASE_URL: str = _require("DATABASE_URL")

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# ── Embedding ─────────────────────────────────────────────────────────────────
# gemini-embedding-001 produces up to 3072-dimensional vectors; we pin to 768
# for a good quality/cost balance.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# ── Gemini LLM ────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Connection pooling ────────────────────────────────────────────────────────
# Supabase (psycopg2 ThreadedConnectionPool)
DB_POOL_MIN: int = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX: int = int(os.getenv("DB_POOL_MAX", "10"))

# Pinecone — controls the urllib3 thread pool used for parallel upsert batches
PINECONE_POOL_THREADS: int = int(os.getenv("PINECONE_POOL_THREADS", "4"))
