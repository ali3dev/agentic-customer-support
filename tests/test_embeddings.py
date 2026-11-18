from src.rag.embeddings import EmbeddingService


class FakeVector:
    """A lightweight replacement for a NumPy embedding vector."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEmbeddingModel:
    """Simulate FastEmbed without downloading or loading a model."""

    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        return iter(
            FakeVector([float(index), 0.5])
            for index, _ in enumerate(texts)
        )

    def query_embed(self, text):
        return iter([FakeVector([0.1, 0.2])])


def create_service(monkeypatch):
    monkeypatch.setattr(
        "src.rag.embeddings.TextEmbedding",
        FakeEmbeddingModel,
    )

    return EmbeddingService(model_name="fake-model")


def test_embedding_service_initializes_model(monkeypatch):
    service = create_service(monkeypatch)

    assert service.model.model_name == "fake-model"


def test_embed_documents_returns_vectors(monkeypatch):
    service = create_service(monkeypatch)

    result = service.embed_documents([
        "Returns are allowed within 30 days.",
        "Refunds take 5 to 7 business days.",
    ])

    assert len(result) == 2
    assert result[0] == [0.0, 0.5]
    assert result[1] == [1.0, 0.5]


def test_embed_query_returns_vector(monkeypatch):
    service = create_service(monkeypatch)

    result = service.embed_query("When will I get my refund?")

    assert result == [0.1, 0.2]


def test_embed_documents_returns_empty_list(monkeypatch):
    service = create_service(monkeypatch)

    assert service.embed_documents([]) == []


def test_embed_documents_rejects_blank_text(monkeypatch):
    service = create_service(monkeypatch)

    try:
        service.embed_documents(["Valid text", "   "])
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "Document texts cannot be empty."


def test_embed_query_rejects_blank_text(monkeypatch):
    service = create_service(monkeypatch)

    try:
        service.embed_query("   ")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "Query text cannot be empty."
