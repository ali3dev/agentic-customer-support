from concurrent.futures import ThreadPoolExecutor

import pytest

from src.rag.retrieval import RetrievalService
from src.tenant_services import TenantServiceRegistry


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def create_registry(tmp_path):
    return TenantServiceRegistry(
        embedding_service=FakeEmbeddingService(),
        data_dir=tmp_path / "data",
        knowledge_base_dir=tmp_path / "knowledge",
    )


def test_failed_update_restores_document_and_vector_inventory(
    tmp_path,
    monkeypatch,
):
    import src.tenant_services as tenant_services_module

    registry = create_registry(tmp_path)
    tenant_id = "tenant-a"
    registry.put_knowledge_document(
        tenant_id,
        "policy.md",
        "# Policy\n\nOriginal verified policy.",
    )
    original_ingest = tenant_services_module.ingest_directory
    fail_once = True

    def fail_first_sync(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated vector-store failure")
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(
        tenant_services_module,
        "ingest_directory",
        fail_first_sync,
    )

    with pytest.raises(RuntimeError, match="simulated vector-store failure"):
        registry.put_knowledge_document(
            tenant_id,
            "policy.md",
            "# Policy\n\nUncommitted replacement.",
        )

    document = registry.knowledge_base_path(tenant_id) / "policy.md"
    assert document.read_text(encoding="utf-8") == (
        "# Policy\n\nOriginal verified policy."
    )
    assert not registry._is_sync_pending(tenant_id)
    service = registry.get_retrieval_service(tenant_id)
    assert any(
        "Original verified policy" in item["text"]
        for item in service.retrieve("policy", top_k=5)
    )


def test_failed_delete_restores_document_and_vector_inventory(
    tmp_path,
    monkeypatch,
):
    import src.tenant_services as tenant_services_module

    registry = create_registry(tmp_path)
    tenant_id = "tenant-a"
    registry.put_knowledge_document(
        tenant_id,
        "policy.md",
        "# Policy\n\nPolicy remains available.",
    )
    original_ingest = tenant_services_module.ingest_directory
    fail_once = True

    def fail_first_sync(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated vector-store failure")
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(
        tenant_services_module,
        "ingest_directory",
        fail_first_sync,
    )

    with pytest.raises(RuntimeError, match="simulated vector-store failure"):
        registry.delete_knowledge_document(tenant_id, "policy.md")

    document = registry.knowledge_base_path(tenant_id) / "policy.md"
    assert document.read_text(encoding="utf-8") == (
        "# Policy\n\nPolicy remains available."
    )
    assert any(
        "Policy remains available" in item["text"]
        for item in registry.get_retrieval_service(tenant_id).retrieve(
            "policy",
            top_k=5,
        )
    )


def test_failed_rollback_leaves_journal_for_later_reconciliation(
    tmp_path,
    monkeypatch,
):
    import src.tenant_services as tenant_services_module

    registry = create_registry(tmp_path)
    tenant_id = "tenant-a"
    registry.put_knowledge_document(
        tenant_id,
        "policy.md",
        "# Policy\n\nOriginal source.",
    )
    original_ingest = tenant_services_module.ingest_directory

    def always_fail(*args, **kwargs):
        raise RuntimeError("simulated persistent vector-store failure")

    monkeypatch.setattr(
        tenant_services_module,
        "ingest_directory",
        always_fail,
    )

    with pytest.raises(RuntimeError, match="recovery is pending"):
        registry.put_knowledge_document(
            tenant_id,
            "policy.md",
            "# Policy\n\nReplacement source.",
        )

    assert registry._is_sync_pending(tenant_id)
    monkeypatch.setattr(
        tenant_services_module,
        "ingest_directory",
        original_ingest,
    )

    service = registry.get_retrieval_service(tenant_id)
    assert not registry._is_sync_pending(tenant_id)
    assert any(
        "Original source" in item["text"]
        for item in service.retrieve("policy", top_k=5)
    )


def test_pending_sync_is_recovered_before_retrieval(tmp_path):
    registry = create_registry(tmp_path)
    tenant_id = "tenant-a"
    root = registry.knowledge_base_path(tenant_id)
    root.mkdir(parents=True)
    (root / "policy.md").write_text(
        "# Policy\n\nRecovered after interrupted sync.",
        encoding="utf-8",
    )
    registry._mark_sync_pending(tenant_id)

    service = registry.get_retrieval_service(tenant_id)

    assert not registry._is_sync_pending(tenant_id)
    assert any(
        "Recovered after interrupted sync" in item["text"]
        for item in service.retrieve("policy", top_k=5)
    )


def test_concurrent_document_updates_leave_index_matching_source(
    tmp_path,
):
    registry = create_registry(tmp_path)
    other_worker = create_registry(tmp_path)
    tenant_id = "tenant-a"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                worker.put_knowledge_document,
                tenant_id,
                "policy.md",
                f"# Policy\n\nConcurrent update {index}.",
            )
            for worker, index in (
                (registry, 0),
                (other_worker, 1),
            )
        ]
        for future in futures:
            future.result()

    document = registry.knowledge_base_path(tenant_id) / "policy.md"
    expected = document.read_text(encoding="utf-8").splitlines()[-1]
    service: RetrievalService = registry.get_retrieval_service(tenant_id)
    results = service.retrieve("policy", top_k=5)

    assert any(expected in item["text"] for item in results)
