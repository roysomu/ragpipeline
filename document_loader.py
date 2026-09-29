"""
document_loader.py — Load and chunk PDF files into text chunks.

Each chunk is returned as a dict:
    {
        "text":      str,   # chunk content
        "source":    str,   # filename
        "page":      int,   # 1-based page number
        "chunk_idx": int,   # chunk index within that page
    }

Supports loading from:
  - a local file path  (load_pdf)
  - raw bytes in memory (load_pdf_from_bytes) — used for Supabase-stored files
"""

import io
import os
from pathlib import Path
from typing import Generator

from pypdf import PdfReader
from tqdm import tqdm

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── helpers ───────────────────────────────────────────────────────────────────

def _sliding_window_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── public API ────────────────────────────────────────────────────────────────

def load_pdf(file_path: str | Path) -> Generator[dict, None, None]:
    """
    Yield text chunks from a single PDF file.

    Args:
        file_path: Path to the PDF.

    Yields:
        Chunk dicts with keys: text, source, page, chunk_idx.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    source_name = path.name

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue

        for idx, chunk in enumerate(_sliding_window_chunks(raw_text)):
            yield {
                "text": chunk,
                "source": source_name,
                "page": page_num,
                "chunk_idx": idx,
            }


def load_pdf_from_bytes(
    data: bytes, filename: str
) -> Generator[dict, None, None]:
    """
    Yield text chunks from raw PDF bytes (e.g. retrieved from Supabase).

    This is the in-memory equivalent of *load_pdf* — no local file needed.

    Args:
        data:     Raw PDF bytes.
        filename: Original filename, used as the *source* label in metadata.

    Yields:
        Chunk dicts with keys: text, source, page, chunk_idx.
    """
    reader = PdfReader(io.BytesIO(data))

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue

        for idx, chunk in enumerate(_sliding_window_chunks(raw_text)):
            yield {
                "text": chunk,
                "source": filename,
                "page": page_num,
                "chunk_idx": idx,
            }


def load_pdfs_from_folder(folder: str | Path) -> list[dict]:

    """
    Load all PDFs from *folder* and return a flat list of chunk dicts.

    Args:
        folder: Directory containing PDF files.

    Returns:
        List of chunk dicts.
    """
    folder = Path(folder)
    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    all_chunks: list[dict] = []
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        chunks = list(load_pdf(pdf_path))
        all_chunks.extend(chunks)
        print(f"  ✓ {pdf_path.name}: {len(chunks)} chunks")

    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks
