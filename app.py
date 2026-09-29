"""
app.py — Gradio UI wrapper for the RAG pipeline.

Run:
    python3 app.py

Then open http://localhost:7860 in your browser.
"""

from pathlib import Path

import gradio as gr

from rag_pipeline import RAGPipeline
import supabase_storage

# ── Shared pipeline instance (created once on startup) ────────────────────────
pipeline = RAGPipeline(verbose=False)


# ── Backend functions ─────────────────────────────────────────────────────────

def ingest_files(files) -> str:
    """Save uploaded PDFs to Supabase and ingest them into Pinecone."""
    if not files:
        return "⚠️ No files uploaded."

    results = []
    for file in files:
        path = Path(file.name)
        try:
            # Read raw bytes from the temp file Gradio created
            data = path.read_bytes()
            filename = path.name

            # 1. Persist to Supabase
            doc_id = supabase_storage.save_document(filename, data)

            # 2. Embed & upsert into Pinecone (directly from bytes — no disk I/O)
            pipeline.ingest_bytes(data, filename)

            # 3. Mark as ingested in the DB
            supabase_storage.mark_ingested(doc_id)

            results.append(
                f"✅ {filename} — saved to Supabase (id={doc_id}) & ingested"
            )
        except Exception as e:
            results.append(f"❌ {path.name} — {e}")

    return "\n".join(results)


def list_documents() -> str:
    """Return a markdown table of all documents stored in Supabase."""
    try:
        rows = supabase_storage.list_documents()
    except Exception as e:
        return f"❌ Could not connect to Supabase: {e}"

    if not rows:
        return "No documents stored yet."

    lines = ["| ID | Filename | Uploaded At | Ingested |", "|----|----------|-------------|----------|"]
    for r in rows:
        ingested = "✅" if r["ingested"] else "⏳"
        lines.append(f"| {r['id']} | {r['filename']} | {r['uploaded_at']} | {ingested} |")
    return "\n".join(lines)


def ask_question(question: str, source_filter: str) -> tuple[str, str]:
    """Run a RAG query and return the answer + formatted context."""
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    source = source_filter.strip() or None

    try:
        answer, context = pipeline.query(
            question, source_filter=source, return_context=True
        )
    except Exception as e:
        return f"❌ Error: {e}", ""

    if not context:
        return answer, ""

    # Format retrieved context chunks
    ctx_lines = []
    for i, c in enumerate(context, 1):
        ctx_lines.append(
            f"**[{i}]** `{c['source']}` — page {c['page']} "
            f"(score: {c['score']:.3f})\n\n{c['text']}\n"
        )
    context_md = "\n---\n".join(ctx_lines)

    return answer, context_md


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG Pipeline") as demo:
    gr.Markdown("# 📚 RAG Pipeline — Supabase + Pinecone + Gemini")
    gr.Markdown(
        "Upload PDFs to store them in Supabase and index them into Pinecone, "
        "then ask questions grounded in their content."
    )

    with gr.Tabs():

        # ── Tab 1: Ingest ──────────────────────────────────────────────────────
        with gr.Tab("📥 Upload & Ingest PDFs"):
            gr.Markdown("### Upload PDF files — stored in Supabase, indexed in Pinecone")
            file_input = gr.File(
                label="Select PDF(s)",
                file_types=[".pdf"],
                file_count="multiple",
            )
            ingest_btn = gr.Button("Upload & Ingest", variant="primary")
            ingest_output = gr.Textbox(
                label="Result", lines=6, interactive=False
            )

            ingest_btn.click(
                fn=ingest_files,
                inputs=file_input,
                outputs=ingest_output,
            )

        # ── Tab 2: Documents ───────────────────────────────────────────────────
        with gr.Tab("🗄️ Stored Documents"):
            gr.Markdown("### All PDF files stored in Supabase")
            refresh_btn = gr.Button("Refresh List", variant="secondary")
            docs_output = gr.Markdown()

            refresh_btn.click(fn=list_documents, inputs=[], outputs=docs_output)
            demo.load(fn=list_documents, inputs=[], outputs=docs_output)

        # ── Tab 3: Query ───────────────────────────────────────────────────────
        with gr.Tab("💬 Ask a Question"):
            gr.Markdown("### Query your indexed documents")
            with gr.Row():
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What are the key findings?",
                    scale=4,
                )
                source_box = gr.Textbox(
                    label="Filter by filename (optional)",
                    placeholder="e.g. report.pdf",
                    scale=1,
                )

            ask_btn = gr.Button("Ask", variant="primary")

            answer_box = gr.Textbox(
                label="Answer", lines=6, interactive=False
            )
            context_box = gr.Markdown(label="Retrieved Context")

            ask_btn.click(
                fn=ask_question,
                inputs=[question_box, source_box],
                outputs=[answer_box, context_box],
            )

            # Also trigger on Enter key in the question box
            question_box.submit(
                fn=ask_question,
                inputs=[question_box, source_box],
                outputs=[answer_box, context_box],
            )


if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Soft())
