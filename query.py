"""
query.py — Interactive RAG query CLI.

Usage:
    python query.py                       # starts an interactive REPL
    python query.py --question "..."      # one-shot query then exits
    python query.py --source report.pdf   # filter to a specific document
"""

import argparse

from embedder import embed_query
from generator import generate_answer
from vector_store import get_or_create_index, query_index


def ask(
    question: str,
    index,
    source_filter: str | None = None,
) -> str:
    """
    Run a single RAG query.

    Args:
        question:      The user's question.
        index:         Pinecone Index object (shared across calls).
        source_filter: If set, restrict retrieval to this source filename.

    Returns:
        Generated answer string.
    """
    query_embedding = embed_query(question)

    pinecone_filter = {"source": {"$eq": source_filter}} if source_filter else None
    context_chunks = query_index(index, query_embedding, filter=pinecone_filter)

    if not context_chunks:
        return "No relevant context found in the index. Have you run ingest.py first?"

    print("\n📄 Retrieved context from:")
    for c in context_chunks:
        print(f"   • {c['source']} | page {c['page']} | score {c['score']:.3f}")

    return generate_answer(question, context_chunks)


def interactive_loop(index, source_filter: str | None = None) -> None:
    """Start an interactive question-answering REPL.

    Args:
        index:         Pinecone Index object (initialised once before the loop).
        source_filter: Optional PDF filename to restrict retrieval to.
    """
    print("RAG Pipeline ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        answer = ask(question, index, source_filter=source_filter)
        print(f"\nAssistant: {answer}\n")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--question", "-q", type=str, help="One-shot question")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter retrieval to a specific PDF filename (e.g. report.pdf)",
    )
    args = parser.parse_args()

    # Initialise the Pinecone index once — shared across all queries.
    index = get_or_create_index()

    if args.question:
        answer = ask(args.question, index, source_filter=args.source)
        print(f"\nAnswer: {answer}")
    else:
        interactive_loop(index, source_filter=args.source)


if __name__ == "__main__":
    main()
