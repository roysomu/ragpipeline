"""
ingest.py — CLI script to index PDF files into Pinecone.

Usage:
    python ingest.py --folder ./docs
    python ingest.py --file ./docs/report.pdf
"""

import argparse
from pathlib import Path

from document_loader import load_pdf, load_pdfs_from_folder
from embedder import embed_texts
from vector_store import get_or_create_index, upsert_chunks


def ingest(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunks to ingest.")
        return

    print(f"\n→ Embedding {len(chunks)} chunks …")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    index = get_or_create_index()
    upsert_chunks(index, chunks, embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into Pinecone.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Folder containing PDF files")
    group.add_argument("--file", type=str, help="Single PDF file to ingest")
    args = parser.parse_args()

    if args.folder:
        chunks = load_pdfs_from_folder(args.folder)
    else:
        chunks = list(load_pdf(args.file))
        print(f"Loaded {len(chunks)} chunks from {Path(args.file).name}")

    ingest(chunks)


if __name__ == "__main__":
    main()
