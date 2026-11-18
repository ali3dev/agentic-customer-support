
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from src.rag.similarity import cosine_similarity


class SQLiteVectorStore:
    """Persist document chunks and embeddings in SQLite."""

    REQUIRED_FIELDS = (
        "chunk_id",
        "document_id",
        "text",
        "embedding",
        "source_path",
        "section_title",
        "ordinal",
    )

    def __init__(
        self,
        db_path: str = "data/vectors.db",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        connection = self._get_connection()

        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        section_title TEXT NOT NULL,
                        ordinal INTEGER NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id)
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS managed_documents (
                        document_id TEXT PRIMARY KEY
                    )
                    """
                )
        finally:
            connection.close()

    @classmethod
    def _prepare_records(
        cls,
        chunks: list[dict[str, Any]],
    ) -> list[tuple]:
        """Validate chunks and serialize them for SQLite."""

        records = []

        for chunk in chunks:
            missing_fields = [
                field
                for field in cls.REQUIRED_FIELDS
                if field not in chunk
            ]

            if missing_fields:
                raise ValueError(
                    f"Missing required fields: {missing_fields}"
                )

            embedding = chunk["embedding"]

            if (
                not isinstance(embedding, (list, tuple))
                or not embedding
            ):
                raise ValueError(
                    "Embedding must be a non-empty list or tuple."
                )

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in embedding
            ):
                raise ValueError(
                    "Embedding must contain finite numeric values."
                )

            records.append(
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["text"],
                    json.dumps(embedding),
                    chunk["source_path"],
                    chunk["section_title"],
                    chunk["ordinal"],
                )
            )

        return records

    @staticmethod
    def _upsert_records(
        connection: sqlite3.Connection,
        records: list[tuple],
    ) -> None:
        """Insert records or update existing chunk IDs."""

        if not records:
            return

        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                document_id,
                text,
                embedding,
                source_path,
                section_title,
                ordinal
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                text = excluded.text,
                embedding = excluded.embedding,
                source_path = excluded.source_path,
                section_title = excluded.section_title,
                ordinal = excluded.ordinal
            """,
            records,
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Insert new chunks or update existing chunks."""

        if not chunks:
            return 0

        records = self._prepare_records(chunks)

        connection = self._get_connection()

        try:
            with connection:
                self._upsert_records(connection, records)
        finally:
            connection.close()

        return len(records)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Atomically replace chunks belonging to one document."""

        if (
            not isinstance(document_id, str)
            or not document_id.strip()
        ):
            raise ValueError(
                "document_id must be a non-empty string."
            )

        records = self._prepare_records(chunks)

        for record in records:
            if record[1] != document_id:
                raise ValueError(
                    "Every chunk must belong to the specified "
                    f"document_id: {document_id}"
                )

        connection = self._get_connection()

        try:
            with connection:
                connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )

                self._upsert_records(connection, records)
        finally:
            connection.close()

        return len(records)

    def sync_managed_documents(
        self,
        documents: dict[str, list[dict[str, Any]]],
        allow_empty: bool = False,
    ) -> dict[str, int]:
        """
        Atomically synchronize all ingestion-managed documents.

        Documents absent from the new inventory are removed.
        Unmanaged chunks are preserved.

        All validation happens before the database transaction.
        """

        if not isinstance(documents, dict):
            raise ValueError("documents must be a dictionary.")

        if not documents and not allow_empty:
            raise ValueError(
                "Refusing to synchronize an empty document inventory. "
                "Pass allow_empty=True to explicitly clear managed data."
            )

        prepared_documents = {}
        all_records = []
        chunk_ids = set()

        for document_id, chunks in documents.items():
            if (
                not isinstance(document_id, str)
                or not document_id.strip()
            ):
                raise ValueError(
                    "Every document_id must be a non-empty string."
                )

            if not isinstance(chunks, list):
                raise ValueError(
                    f"Chunks for {document_id} must be a list."
                )

            records = self._prepare_records(chunks)

            for record in records:
                if record[1] != document_id:
                    raise ValueError(
                        f"Chunk does not belong to document: "
                        f"{document_id}"
                    )

                chunk_id = record[0]

                if chunk_id in chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk_id in inventory: {chunk_id}"
                    )

                chunk_ids.add(chunk_id)

            prepared_documents[document_id] = records
            all_records.extend(records)

        connection = self._get_connection()

        try:
            with connection:
                existing_managed = connection.execute(
                    "SELECT document_id FROM managed_documents"
                ).fetchall()

                existing_ids = {
                    row["document_id"]
                    for row in existing_managed
                }

                # Prevent managed chunks from overwriting
                # unrelated, manually inserted chunks.
                for record in all_records:
                    existing = connection.execute(
                        """
                        SELECT document_id
                        FROM chunks
                        WHERE chunk_id = ?
                        """,
                        (record[0],),
                    ).fetchone()

                    if (
                            existing is not None
                            and existing["document_id"] not in existing_ids
                            and existing["document_id"] != record[1]
                        ):
                        raise ValueError(
                            "Chunk ID conflicts with an unmanaged "
                            f"record: {record[0]}"
                        )

                # Remove previously managed documents.
                connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id IN (
                        SELECT document_id
                        FROM managed_documents
                    )
                    """
                )

                # Adopt and replace any existing records belonging to
                # documents in the current authoritative inventory.
                for document_id in prepared_documents:
                    connection.execute(
                        "DELETE FROM chunks WHERE document_id = ?",
                        (document_id,),
                    )

                # Insert the complete new inventory.
                if all_records:
                    connection.executemany(
                        """
                        INSERT INTO chunks (
                            chunk_id,
                            document_id,
                            text,
                            embedding,
                            source_path,
                            section_title,
                            ordinal
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        all_records,
                    )

                # Replace the ownership manifest.
                connection.execute(
                    "DELETE FROM managed_documents"
                )

                if prepared_documents:
                    connection.executemany(
                        """
                        INSERT INTO managed_documents (document_id)
                        VALUES (?)
                        """,
                        [
                            (document_id,)
                            for document_id in prepared_documents
                        ],
                    )

        finally:
            connection.close()

        removed_documents = len(
            existing_ids - set(prepared_documents)
        )

        return {
            "chunks_stored": len(all_records),
            "documents_removed": removed_documents,
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks for a query embedding."""

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

        cosine_similarity(query_embedding, query_embedding)

        connection = self._get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    text,
                    embedding,
                    source_path,
                    section_title,
                    ordinal
                FROM chunks
                ORDER BY ordinal ASC, chunk_id ASC
                """
            ).fetchall()
        finally:
            connection.close()

        results = []

        for row in rows:
            document_embedding = json.loads(row["embedding"])

            score = cosine_similarity(
                query_embedding,
                document_embedding,
            )

            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "source_path": row["source_path"],
                    "section_title": row["section_title"],
                    "ordinal": row["ordinal"],
                    "score": score,
                }
            )

        results.sort(
            key=lambda result: (
                -result["score"],
                result["ordinal"],
                result["chunk_id"],
            )
        )

        return results[:top_k]
