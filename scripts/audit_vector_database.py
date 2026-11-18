
import sqlite3

from src.rag.vector_store import SQLiteVectorStore


def main() -> None:
    vector_store = SQLiteVectorStore(
        db_path="data/vectors.db"
    )

    connection = vector_store._get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                document_id,
                source_path,
                COUNT(*) AS chunk_count
            FROM chunks
            GROUP BY document_id, source_path
            ORDER BY document_id
            """
        ).fetchall()

        print("VECTOR DATABASE AUDIT")
        print("=" * 70)

        if not rows:
            print("No chunks found.")
            return

        for row in rows:
            print(f"\nDocument ID: {row['document_id']}")
            print(f"Source: {row['source_path']}")
            print(f"Chunks: {row['chunk_count']}")

        print("\n" + "=" * 70)

        total = connection.execute(
            "SELECT COUNT(*) AS total FROM chunks"
        ).fetchone()

        print(f"Total chunks: {total['total']}")

        print("\nINDIVIDUAL CHUNK IDs")
        print("=" * 70)

        chunks = connection.execute(
            """
            SELECT
                chunk_id,
                document_id,
                source_path,
                section_title
            FROM chunks
            ORDER BY document_id, ordinal
            """
        ).fetchall()

        for chunk in chunks:
            print(
                f"{chunk['chunk_id']} | "
                f"{chunk['document_id']} | "
                f"{chunk['source_path']} | "
                f"{chunk['section_title']}"
            )

        print("\nSUSPICIOUS DOCUMENT CONTENT")
        print("=" * 70)

        legacy_chunks = connection.execute(
            """
            SELECT chunk_id, text
            FROM chunks
            WHERE document_id = ?
            ORDER BY ordinal
            """,
            ("doc-001",),
        ).fetchall()

        for chunk in legacy_chunks:
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Text: {chunk['text']}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
