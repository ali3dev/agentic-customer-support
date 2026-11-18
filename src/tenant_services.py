import logging
import hashlib
import os
import re
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from requests.exceptions import RequestException
from src.rag.embeddings import EmbeddingService
from src.rag.ingestion import ingest_directory
from src.rag.retrieval import (
    DEFAULT_RELEVANCE_THRESHOLD,
    RetrievalService,
)
from src.rag.vector_store import SQLiteVectorStore


logger = logging.getLogger(__name__)
_SYNC_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    sqlite3.Error,
    RequestException,
)


class TenantServiceRegistry:
    """Create isolated vector stores and knowledge-base paths per tenant."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        data_dir: str | Path,
        knowledge_base_dir: str | Path,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        self.embedding_service = embedding_service
        self.data_dir = Path(data_dir)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.relevance_threshold = relevance_threshold
        self._services: dict[str, tuple[RetrievalService, SQLiteVectorStore]] = {}
        self._lock = threading.Lock()
        self._lock_db_path = self.data_dir / "knowledge_base_locks.db"
        self._journal_db_path = self.data_dir / "knowledge_base_sync.db"
        self._initialize_sync_journal()

    def _initialize_sync_journal(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._journal_db_path, timeout=30)
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pending_sync (
                        tenant_id TEXT PRIMARY KEY
                    )
                    """
                )
        finally:
            connection.close()

    @contextmanager
    def _exclusive_sync(self) -> Generator[None, None, None]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._lock_db_path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _is_sync_pending(self, tenant_id: str) -> bool:
        connection = sqlite3.connect(self._journal_db_path, timeout=30)
        try:
            return connection.execute(
                "SELECT 1 FROM pending_sync WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone() is not None
        finally:
            connection.close()

    def _mark_sync_pending(self, tenant_id: str) -> None:
        connection = sqlite3.connect(self._journal_db_path, timeout=30)
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO pending_sync (tenant_id) VALUES (?)",
                    (tenant_id,),
                )
        finally:
            connection.close()

    def _clear_sync_pending(self, tenant_id: str) -> None:
        connection = sqlite3.connect(self._journal_db_path, timeout=30)
        try:
            with connection:
                connection.execute(
                    "DELETE FROM pending_sync WHERE tenant_id = ?",
                    (tenant_id,),
                )
        finally:
            connection.close()

    @staticmethod
    def _tenant_directory_name(tenant_id: str) -> str:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string.")

        return hashlib.sha256(
            tenant_id.encode("utf-8")
        ).hexdigest()

    def knowledge_base_path(self, tenant_id: str) -> Path:
        return (
            self.knowledge_base_dir
            / self._tenant_directory_name(tenant_id)
        )

    def _get_or_create_retrieval_service(
        self,
        tenant_id: str,
    ) -> RetrievalService:
        with self._lock:
            if tenant_id not in self._services:
                tenant_directory = self._tenant_directory_name(tenant_id)
                vector_store = SQLiteVectorStore(
                    db_path=str(
                        self.data_dir
                        / "tenant_vectors"
                        / tenant_directory
                        / "vectors.db"
                    )
                )
                retrieval_service = RetrievalService(
                    embedding_service=self.embedding_service,
                    vector_store=vector_store,
                    relevance_threshold=self.relevance_threshold,
                )
                self._services[tenant_id] = (
                    retrieval_service,
                    vector_store,
                )

            return self._services[tenant_id][0]

    def get_retrieval_service(self, tenant_id: str) -> RetrievalService:
        with self._exclusive_sync():
            retrieval_service = self._get_or_create_retrieval_service(
                tenant_id
            )
            if self._is_sync_pending(tenant_id):
                self._mark_sync_pending(tenant_id)
                self._sync_knowledge_base_locked(tenant_id)
                self._clear_sync_pending(tenant_id)
            return retrieval_service

    def _sync_knowledge_base_locked(self, tenant_id: str) -> dict[str, int]:
        self._get_or_create_retrieval_service(tenant_id)
        vector_store = self._services[tenant_id][1]
        return ingest_directory(
            knowledge_base_path=self.knowledge_base_path(tenant_id),
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            allow_empty=True,
        )

    def sync_knowledge_base(self, tenant_id: str) -> dict[str, int]:
        with self._exclusive_sync():
            self._mark_sync_pending(tenant_id)
            result = self._sync_knowledge_base_locked(tenant_id)
            self._clear_sync_pending(tenant_id)
            return result

    @staticmethod
    def _validate_document_name(document_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.md", document_name):
            raise ValueError("Invalid knowledge-base document name.")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".kb-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _restore_document(
        self,
        document_path: Path,
        old_content: str | None,
    ) -> None:
        if old_content is None:
            document_path.unlink(missing_ok=True)
        else:
            self._atomic_write(document_path, old_content)

    def _rollback_document_change(
        self,
        tenant_id: str,
        document_path: Path,
        old_content: str | None,
    ) -> None:
        try:
            self._restore_document(document_path, old_content)
            self._sync_knowledge_base_locked(tenant_id)
        except _SYNC_ERRORS:
            logger.exception(
                "Knowledge-base reconciliation remains pending.",
                extra={"tenant_id": tenant_id},
            )
            raise RuntimeError(
                "Knowledge-base recovery is pending; retrieval will retry."
            ) from None
        self._clear_sync_pending(tenant_id)

    def put_knowledge_document(
        self,
        tenant_id: str,
        document_name: str,
        content: str,
    ) -> dict[str, int]:
        self._validate_document_name(document_name)
        root = self.knowledge_base_path(tenant_id)
        root.mkdir(parents=True, exist_ok=True)
        document_path = root / document_name

        with self._exclusive_sync():
            if document_path.is_symlink():
                raise ValueError("Symbolic links are not allowed.")
            old_content = (
                document_path.read_text(encoding="utf-8")
                if document_path.is_file()
                else None
            )
            self._mark_sync_pending(tenant_id)
            try:
                self._atomic_write(document_path, content)
                result = self._sync_knowledge_base_locked(tenant_id)
            except _SYNC_ERRORS:
                self._rollback_document_change(
                    tenant_id,
                    document_path,
                    old_content,
                )
                raise
            self._clear_sync_pending(tenant_id)
            return result

    def delete_knowledge_document(
        self,
        tenant_id: str,
        document_name: str,
    ) -> dict[str, int]:
        self._validate_document_name(document_name)
        document_path = self.knowledge_base_path(tenant_id) / document_name

        with self._exclusive_sync():
            if document_path.is_symlink():
                raise ValueError("Symbolic links are not allowed.")
            if not document_path.is_file():
                raise FileNotFoundError(document_name)

            old_content = document_path.read_text(encoding="utf-8")
            self._mark_sync_pending(tenant_id)
            try:
                document_path.unlink()
                result = self._sync_knowledge_base_locked(tenant_id)
            except _SYNC_ERRORS:
                self._rollback_document_change(
                    tenant_id,
                    document_path,
                    old_content,
                )
                raise
            self._clear_sync_pending(tenant_id)
            return result
