
from pathlib import Path

from src.rag.chunking import chunk_markdown
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import SQLiteVectorStore


def ingest_directory(
    knowledge_base_path: str | Path,
    embedding_service: EmbeddingService,
    vector_store: SQLiteVectorStore,
    max_chars: int = 1500,
    overlap: int = 150,
    allow_empty: bool = False,
) -> dict[str, int]:
    """
    Synchronize a Markdown knowledge base with SQLite.

    All files are read, chunked and embedded before the
    database is modified.
    """

    root = Path(knowledge_base_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {root}"
        )

    markdown_files = sorted(root.rglob("*.md"))

    if not markdown_files and not allow_empty:
        raise ValueError(
            "Knowledge base contains no Markdown files. "
            "Pass allow_empty=True to explicitly clear managed documents."
        )

    documents: dict[str, list[dict]] = {}

    documents_ingested = 0
    chunks_created = 0

    # Phase 1: Prepare the entire inventory outside SQLite.
    for file_path in markdown_files:
        markdown = file_path.read_text(encoding="utf-8")

        relative_path = file_path.relative_to(root).as_posix()

        chunks = chunk_markdown(
            markdown=markdown,
            document_id=relative_path,
            source_path=relative_path,
            max_chars=max_chars,
            overlap=overlap,
        )

        documents[relative_path] = chunks

        if not chunks:
            continue

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_service.embed_documents(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count: "
                f"{len(embeddings)} embeddings for "
                f"{len(chunks)} chunks."
            )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        documents_ingested += 1
        chunks_created += len(chunks)

    # Phase 2: Atomically synchronize the complete inventory.
    sync_result = vector_store.sync_managed_documents(
        documents=documents,
        allow_empty=allow_empty,
    )

    return {
        "documents_found": len(markdown_files),
        "documents_ingested": documents_ingested,
        "documents_removed": sync_result["documents_removed"],
        "chunks_created": chunks_created,
        "chunks_stored": sync_result["chunks_stored"],
    }
