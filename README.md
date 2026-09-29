# RAG Pipeline with Pinecone + Supabase + Gemini

A production-ready Retrieval-Augmented Generation (RAG) pipeline with a **Gradio web UI**. Upload PDFs to **Supabase** (PostgreSQL for persistent storage), index them into **Pinecone** (managed serverless vector DB) using **Google `gemini-embedding-001`** embeddings, and answer questions with **Google Gemini**, grounded in retrieved context.

## Architecture

```
PDF Upload (Gradio UI)
   │
   ├──► supabase_storage.py  ←  persist raw PDF bytes in PostgreSQL (rag_documents table)
   │
   ▼
document_loader.py           ←  sliding-window chunking (words)
   │
   ▼
embedder.py                  ←  Google gemini-embedding-001 (768-dim)
   │
   ▼
vector_store.py              ←  Pinecone serverless (upsert / query)
   │           ▲
   │           │  (top-k retrieval)
   ▼           │
generator.py  ─────────────  Google Gemini (grounded answer + citations)
   │
   ▼
app.py (Gradio UI)           ←  Upload • Browse • Ask tabs
```

## Project Structure

```
RAG with PC/
├── config.py              # All settings loaded from .env
├── document_loader.py     # PDF → text chunks (sliding window)
├── embedder.py            # Google gemini-embedding-001 embeddings
├── vector_store.py        # Pinecone CRUD wrapper
├── supabase_storage.py    # Supabase/PostgreSQL document store
├── generator.py           # Gemini answer generation
├── rag_pipeline.py        # High-level RAG API
├── ingest.py              # CLI: index PDFs from disk
├── query.py               # CLI: interactive / one-shot QA
├── app.py                 # Gradio web UI
├── requirements.txt
├── .env
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file (or copy and edit the one already present):

| Variable | Where to get it |
|---|---|
| `DATABASE_URL` | [supabase.com](https://supabase.com) → Project Settings → Database → Connection string |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

### 3. Launch the Gradio web UI

```bash
python3 app.py
```

Then open **http://localhost:7860** in your browser.

The UI has three tabs:

| Tab | What it does |
|---|---|
| 📥 **Upload & Ingest PDFs** | Upload one or more PDFs — stored in Supabase and indexed in Pinecone |
| 🗄️ **Stored Documents** | Browse all files saved in Supabase with ingestion status |
| 💬 **Ask a Question** | Ask questions; get a Gemini answer plus the retrieved context chunks |

## Programmatic API

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest a PDF from raw bytes (as received from Supabase)
pipeline.ingest_bytes(pdf_bytes, "report.pdf")

# Ask a question
answer = pipeline.query("What are the key findings?")
print(answer)

# Ask and get retrieved context chunks too
answer, context = pipeline.query(
    "Summarise the methodology",
    return_context=True
)
```

## Configuration Reference

All settings live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | Supabase PostgreSQL connection string |
| `PINECONE_API_KEY` | *(required)* | Pinecone API key |
| `PINECONE_INDEX_NAME` | `rag-index` | Pinecone index name |
| `GOOGLE_API_KEY` | *(required)* | Google API key (embeddings + Gemini) |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Google embedding model |
| `EMBEDDING_DIMENSION` | `768` | Must match the embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for answer generation |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |

## Supabase Schema

The table is created automatically on first use:

```sql
CREATE TABLE IF NOT EXISTS rag_documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT        NOT NULL,
    content     BYTEA       NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested    BOOLEAN     NOT NULL DEFAULT FALSE
);
```

## Notes

- **No OpenAI dependency** — both embeddings and generation use the single `google-genai` SDK.
- The Pinecone index is created automatically on first run as a **Serverless** index on AWS `us-east-1`. Change `cloud`/`region` in `vector_store.py → get_or_create_index()` if needed.
- Chunks are deduplicated via deterministic MD5 IDs — re-ingesting the same file will overwrite existing vectors, not duplicate them.
- The Gemini prompt is grounded: it will refuse to answer if the retrieved context doesn't contain the answer, reducing hallucinations.
- The Gradio UI uploads PDFs **in-memory** (no temporary disk writes) and marks each document as `ingested=TRUE` in Supabase only after Pinecone upsert succeeds.
