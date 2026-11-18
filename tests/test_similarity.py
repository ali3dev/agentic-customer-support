import pytest

from src.rag.similarity import cosine_similarity


def test_identical_vectors_return_one():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_orthogonal_vectors_return_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_opposite_vectors_return_negative_one():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_different_dimensions_raise_error():
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_zero_vector_raises_error():
    with pytest.raises(ValueError, match="Zero vectors"):
        cosine_similarity([0, 0], [1, 0])


def test_empty_vector_raises_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        cosine_similarity([], [1, 0])


def test_non_numeric_values_raise_error():
    with pytest.raises(ValueError, match="finite numeric"):
        cosine_similarity([1, "invalid"], [1, 2])


def test_boolean_values_raise_error():
    with pytest.raises(ValueError, match="finite numeric"):
        cosine_similarity([1, True], [1, 2])