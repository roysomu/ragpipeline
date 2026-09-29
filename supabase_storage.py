"""
supabase_storage.py — Store and retrieve uploaded PDF files in Supabase (PostgreSQL).

The table schema is created automatically on first use:

    CREATE TABLE IF NOT EXISTS rag_documents (
        id          SERIAL PRIMARY KEY,
        filename    TEXT NOT NULL,
        content     BYTEA NOT NULL,
        uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        ingested    BOOLEAN NOT NULL DEFAULT FALSE
    );

Usage:
    from supabase_storage import save_document, list_documents, load_document_bytes

    # Save an uploaded file
    doc_id = save_document("report.pdf", open("report.pdf", "rb").read())

    # List all stored files
    rows = list_documents()

    # Load raw bytes back out
    filename, data = load_document_bytes(doc_id)
"""

import io
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from config import DATABASE_URL, DB_POOL_MIN, DB_POOL_MAX


# ── connection pool (module-level singleton) ──────────────────────────────────

_pool: ThreadedConnectionPool = ThreadedConnectionPool(
    DB_POOL_MIN,
    DB_POOL_MAX,
    dsn=DATABASE_URL,
)


@contextmanager
def _get_conn():
    """
    Borrow a connection from the pool and return it when done.

    Usage::

        with _get_conn() as conn:
            with conn.cursor() as cur:
                ...
    """
    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── schema bootstrap ──────────────────────────────────────────────────────────

def ensure_table() -> None:
    """Create the *rag_documents* table if it does not already exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS rag_documents (
        id          SERIAL PRIMARY KEY,
        filename    TEXT          NOT NULL,
        content     BYTEA         NOT NULL,
        uploaded_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
        ingested    BOOLEAN       NOT NULL DEFAULT FALSE
    );
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


# ── public API ────────────────────────────────────────────────────────────────

def save_document(filename: str, data: bytes) -> int:
    """
    Store a PDF file's raw bytes in Supabase and return its row ID.

    Args:
        filename: Original filename (e.g. "report.pdf").
        data:     Raw PDF bytes.

    Returns:
        The auto-generated integer ID of the inserted row.
    """
    ensure_table()
    sql = """
    INSERT INTO rag_documents (filename, content)
    VALUES (%s, %s)
    RETURNING id;
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (filename, psycopg2.Binary(data)))
            row_id: int = cur.fetchone()[0]
        conn.commit()
    return row_id


def list_documents() -> list[dict]:
    """
    Return metadata for all stored documents (no binary content).

    Returns:
        List of dicts with keys: id, filename, uploaded_at, ingested.
    """
    ensure_table()
    sql = "SELECT id, filename, uploaded_at, ingested FROM rag_documents ORDER BY uploaded_at DESC;"
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def load_document_bytes(doc_id: int) -> tuple[str, bytes]:
    """
    Retrieve a document's filename and raw bytes by ID.

    Args:
        doc_id: Row ID in the documents table.

    Returns:
        (filename, raw_bytes) tuple.

    Raises:
        FileNotFoundError: If no row with *doc_id* exists.
    """
    ensure_table()
    sql = "SELECT filename, content FROM rag_documents WHERE id = %s;"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id,))
            row = cur.fetchone()
    if row is None:
        raise FileNotFoundError(f"Document with id={doc_id} not found in Supabase.")
    filename, content = row
    # psycopg2 returns BYTEA as memoryview — convert to bytes
    return filename, bytes(content)


def mark_ingested(doc_id: int) -> None:
    """
    Set ingested=TRUE for a document row after its chunks have been embedded.

    Args:
        doc_id: Row ID to update.
    """
    sql = "UPDATE rag_documents SET ingested = TRUE WHERE id = %s;"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id,))
        conn.commit()


def get_pending_documents() -> list[dict]:
    """
    Return metadata for documents that have NOT been ingested yet.

    Returns:
        List of dicts with keys: id, filename, uploaded_at.
    """
    ensure_table()
    sql = """
    SELECT id, filename, uploaded_at
    FROM rag_documents
    WHERE ingested = FALSE
    ORDER BY uploaded_at ASC;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]
