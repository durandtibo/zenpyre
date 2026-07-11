from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from zenpyre.documents.fake import generate_fake_documents
from zenpyre.testing.fixtures import faker_available

MODULE = "zenpyre.documents.fake"

#############################################
#     Tests for generate_fake_documents     #
#############################################


# --- basic generation ---


@faker_available
def test_generate_fake_documents_default_count() -> None:
    docs = generate_fake_documents()
    assert len(docs) == 5


@faker_available
def test_generate_fake_documents_custom_count() -> None:
    docs = generate_fake_documents(n=10)
    assert len(docs) == 10


@faker_available
def test_generate_fake_documents_zero_returns_empty_list() -> None:
    assert generate_fake_documents(n=0) == []


@faker_available
def test_generate_fake_documents_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"'n' must be non-negative"):
        generate_fake_documents(n=-1)


@faker_available
def test_generate_fake_documents_returns_document_instances() -> None:
    docs = generate_fake_documents(n=3)
    assert all(isinstance(doc, Document) for doc in docs)


# --- ids ---


@faker_available
def test_generate_fake_documents_ids_are_sequential() -> None:
    docs = generate_fake_documents(n=3)
    assert [doc.id for doc in docs] == ["doc-0", "doc-1", "doc-2"]


@faker_available
def test_generate_fake_documents_ids_are_unique() -> None:
    docs = generate_fake_documents(n=20)
    ids = [doc.id for doc in docs]
    assert len(ids) == len(set(ids))


# --- content ---


@faker_available
def test_generate_fake_documents_page_content_is_nonempty_string() -> None:
    docs = generate_fake_documents(n=3, seed=1)
    assert all(isinstance(doc.page_content, str) and doc.page_content for doc in docs)


@faker_available
def test_generate_fake_documents_nb_sentences_is_passed_through() -> None:
    with patch(f"{MODULE}.faker.Faker") as mock_faker_cls:
        mock_fake = mock_faker_cls.return_value
        mock_fake.paragraph.return_value = "some text"
        mock_fake.name.return_value = "Jane Doe"
        mock_fake.word.return_value = "topic"

        generate_fake_documents(n=1, nb_sentences=8)

        mock_fake.paragraph.assert_called_once_with(nb_sentences=8)


# --- metadata ---


@faker_available
def test_generate_fake_documents_metadata_has_author_and_topic() -> None:
    docs = generate_fake_documents(n=3)
    for doc in docs:
        assert "author" in doc.metadata
        assert "topic" in doc.metadata
        assert isinstance(doc.metadata["author"], str)
        assert isinstance(doc.metadata["topic"], str)


@faker_available
def test_generate_fake_documents_metadata_does_not_contain_extra_keys() -> None:
    docs = generate_fake_documents(n=1)
    assert set(docs[0].metadata.keys()) == {"author", "topic"}


# --- seeding / determinism ---


@faker_available
def test_generate_fake_documents_same_seed_same_output() -> None:
    docs1 = generate_fake_documents(n=5, seed=42)
    docs2 = generate_fake_documents(n=5, seed=42)
    assert [d.page_content for d in docs1] == [d.page_content for d in docs2]
    assert [d.metadata for d in docs1] == [d.metadata for d in docs2]


@faker_available
def test_generate_fake_documents_different_seed_likely_different_output() -> None:
    """Not a strict guarantee (Faker output could theoretically
    coincide), but with two different seeds and multi-sentence
    paragraphs, an accidental collision across the whole batch is
    astronomically unlikely, so this is a reliable regression check."""
    docs1 = generate_fake_documents(n=5, seed=1)
    docs2 = generate_fake_documents(n=5, seed=2)
    assert [d.page_content for d in docs1] != [d.page_content for d in docs2]


@faker_available
def test_generate_fake_documents_no_seed_calls_do_not_raise() -> None:
    # Sanity check that omitting `seed` (the common case) works at all.
    docs = generate_fake_documents(n=2, seed=None)
    assert len(docs) == 2


@faker_available
def test_generate_fake_documents_seed_is_forwarded_to_faker_seed() -> None:
    with patch(f"{MODULE}.faker.Faker") as mock_faker_cls:
        mock_fake = mock_faker_cls.return_value
        mock_fake.paragraph.return_value = "text"
        mock_fake.name.return_value = "name"
        mock_fake.word.return_value = "word"

        generate_fake_documents(n=1, seed=123)

        mock_faker_cls.seed.assert_called_once_with(123)


@faker_available
def test_generate_fake_documents_no_seed_does_not_call_faker_seed() -> None:
    with patch(f"{MODULE}.faker.Faker") as mock_faker_cls:
        mock_fake = mock_faker_cls.return_value
        mock_fake.paragraph.return_value = "text"
        mock_fake.name.return_value = "name"
        mock_fake.word.return_value = "word"

        generate_fake_documents(n=1, seed=None)

        mock_faker_cls.seed.assert_not_called()


# --- optional dependency guard ---


@faker_available
def test_generate_fake_documents_raises_if_faker_unavailable() -> None:
    with (
        patch(f"{MODULE}.check_faker", side_effect=RuntimeError),
        pytest.raises(RuntimeError),
    ):
        generate_fake_documents(n=1)


@faker_available
def test_generate_fake_documents_checks_availability_before_validating_n() -> None:
    """The availability check should happen unconditionally, even for
    inputs (like a negative n) that would otherwise short-circuit with a
    different error — so callers always get an accurate diagnosis of a
    missing dependency rather than a misleading ValueError."""
    with patch(f"{MODULE}.check_faker", side_effect=RuntimeError("missing")) as mock_check:
        with pytest.raises(RuntimeError, match="missing"):
            generate_fake_documents(n=-1)
        mock_check.assert_called_once()
