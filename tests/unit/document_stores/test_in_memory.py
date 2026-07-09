from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.documents import Document

from zenpyre.document_stores import InMemoryDocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            id="1",
            page_content="Intro to Python",
            metadata={"author": "Alice", "year": 2022, "category": "Programming"},
        ),
        Document(
            id="2",
            page_content="Advanced Python",
            metadata={"author": "Alice", "year": 2023, "category": "Programming"},
        ),
        Document(
            id="3",
            page_content="History of Rome",
            metadata={"author": "Bob", "year": 2021, "category": "History"},
        ),
        Document(
            id="4",
            page_content="History of Greece",
            metadata={"author": "Bob", "year": 2020, "category": "History"},
        ),
    ]


##################################################
#     Tests for InMemoryDocumentStore            #
##################################################


# --- repr/str ---


def test_repr(store: InMemoryDocumentStore) -> None:
    assert repr(store).startswith("InMemoryDocumentStore(")


def test_str(store: InMemoryDocumentStore) -> None:
    assert repr(store).startswith("InMemoryDocumentStore(")


# --- add_documents ---


def test_add_documents_increases_count(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


def test_add_documents_no_id_raises(store: InMemoryDocumentStore) -> None:
    with pytest.raises(ValueError, match=r"id"):
        store.add_documents([Document(page_content="No id")])


def test_add_documents_upsert_replaces_existing(store: InMemoryDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Original", metadata={})])
    store.add_documents([Document(id="1", page_content="Updated", metadata={})])
    assert store.count() == 1
    assert store.get("1").page_content == "Updated"


def test_add_documents_empty(store: InMemoryDocumentStore) -> None:
    store.add_documents([])


# --- count ---


def test_count_empty_store(store: InMemoryDocumentStore) -> None:
    assert store.count() == 0


def test_count_after_adding(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.count() == len(docs)


# --- get ---


def test_get_existing_document(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.get("1") == docs[0]


def test_get_missing_document_returns_none(store: InMemoryDocumentStore) -> None:
    assert store.get("nonexistent") is None


def test_get_round_trips_metadata(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.get("1").metadata == docs[0].metadata


# --- get_many ---


def test_get_many_returns_correct_length(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    assert len(store.get_many(["1", "2", "99"])) == 3


def test_get_many_returns_none_for_missing(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.get_many(["1", "99", "2"])
    assert result[1] is None


def test_get_many_preserves_order(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.get_many(["3", "1", "2"])
    assert [r.id for r in result] == ["3", "1", "2"]


# --- filter ---


def test_filter_no_args_returns_all(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert len(store.filter()) == len(docs)


def test_filter_single_field(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Alice")
    assert all(r.metadata["author"] == "Alice" for r in result)
    assert len(result) == 2


def test_filter_multiple_fields(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Alice", category="Programming")
    assert len(result) == 2


def test_filter_no_match_returns_empty(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert store.filter(author="Charlie") == []


def test_filter_preserves_full_document(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.filter(author="Bob", category="History")
    assert all(
        r.metadata == d.metadata
        for r, d in zip(
            sorted(result, key=lambda x: x.id),
            sorted([d for d in docs if d.metadata["author"] == "Bob"], key=lambda x: x.id),
        )
    )


def test_filter_empty_store_returns_empty(store: InMemoryDocumentStore) -> None:
    assert store.filter(author="Alice") == []


# --- delete ---


def test_delete_removes_document(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete("1")
    assert store.count() == len(docs) - 1
    assert store.get("1") is None


def test_delete_nonexistent_is_silent(store: InMemoryDocumentStore) -> None:
    store.delete("nonexistent")


# --- delete_many ---


def test_delete_many_removes_documents(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.count() == len(docs) - 2
    assert store.get("1") is None
    assert store.get("3") is None


def test_delete_many_preserves_other_documents(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many(["1", "3"])
    assert store.get("2") is not None
    assert store.get("4") is not None


def test_delete_many_empty_list_is_no_op(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    store.delete_many([])
    assert store.count() == len(docs)


def test_delete_many_nonexistent_ids_are_silent(store: InMemoryDocumentStore) -> None:
    store.delete_many(["99", "100"])


def test_delete_many_single_id(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    store.delete_many(["2"])
    assert store.count() == len(docs) - 1
    assert store.get("2") is None


# --- check_ids ---


def test_check_ids_all_found(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "2", "3", "4"])
    assert found == ["1", "2", "3", "4"]
    assert missing == []


def test_check_ids_all_missing(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["99", "100"])
    assert found == []
    assert missing == ["99", "100"]


def test_check_ids_mixed(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["1", "99", "3", "42"])
    assert found == ["1", "3"]
    assert missing == ["99", "42"]


def test_check_ids_preserves_order(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    found, missing = store.check_ids(["3", "99", "1", "42", "2"])
    assert found == ["3", "1", "2"]
    assert missing == ["99", "42"]


def test_check_ids_empty_input_returns_empty_lists(store: InMemoryDocumentStore) -> None:
    found, missing = store.check_ids([])
    assert found == []
    assert missing == []


def test_check_ids_empty_store_returns_all_missing(store: InMemoryDocumentStore) -> None:
    found, missing = store.check_ids(["1", "2"])
    assert found == []
    assert missing == ["1", "2"]


def test_check_ids_returns_tuple_of_two_lists(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = store.check_ids(["1", "99"])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# --- all ---


def test_all_empty_store(store: InMemoryDocumentStore) -> None:
    assert store.all() == []


def test_all_returns_all_documents(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = store.all()
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


# --- lazy_all ---


def test_lazy_all_empty_store_yields_nothing(store: InMemoryDocumentStore) -> None:
    assert list(store.lazy_all()) == []


def test_lazy_all_returns_generator(store: InMemoryDocumentStore) -> None:
    result = store.lazy_all()
    assert isinstance(result, Iterator)


def test_lazy_all_yields_one_document_at_a_time(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    gen = store.lazy_all()
    first = next(gen)
    assert isinstance(first, Document)


def test_lazy_all_returns_all_documents(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


def test_lazy_all_matches_all(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    assert list(store.lazy_all()) == store.all()


def test_lazy_all_yields_document_instances(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = list(store.lazy_all())
    assert all(isinstance(doc, Document) for doc in result)


def test_lazy_all_preserves_metadata(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    result = {doc.id: doc for doc in store.lazy_all()}
    for doc in docs:
        assert result[doc.id].metadata == doc.metadata


def test_lazy_all_does_not_mutate_store(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    list(store.lazy_all())
    assert store.count() == len(docs)


def test_lazy_all_single_document(store: InMemoryDocumentStore) -> None:
    store.add_documents([Document(id="1", page_content="Solo", metadata={})])
    result = list(store.lazy_all())
    assert len(result) == 1
    assert result[0].id == "1"


def test_lazy_all_is_lazy_not_exhausted_on_creation(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    """Calling lazy_all() should not itself execute the query eagerly
    consuming results before iteration begins; adding docs after
    creating the generator but before the first next() call should still
    be reflected, confirming the query executes lazily."""
    gen = store.lazy_all()
    store.add_documents(docs)
    result = list(gen)
    assert len(result) == len(docs)


def test_lazy_all_independent_generators_do_not_interfere(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    """Two separate lazy_all() generators should each independently
    yield the full set of documents, since each uses its own cursor."""
    store.add_documents(docs)
    gen1 = store.lazy_all()
    gen2 = store.lazy_all()
    first_from_gen1 = next(gen1)
    result2 = list(gen2)
    remaining_from_gen1 = list(gen1)

    assert len(result2) == len(docs)
    assert len(remaining_from_gen1) + 1 == len(docs)
    assert first_from_gen1.id not in {d.id for d in remaining_from_gen1}


# --- iter_batches ---


def test_iter_batches_empty_store_yields_nothing(store: InMemoryDocumentStore) -> None:
    assert list(store.iter_batches()) == []


def test_iter_batches_returns_generator(store: InMemoryDocumentStore) -> None:
    result = store.iter_batches()
    assert isinstance(result, Iterator)


def test_iter_batches_default_batch_size(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches())
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


def test_iter_batches_yields_correct_batch_sizes(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert [len(b) for b in batches] == [2, 2]


def test_iter_batches_last_batch_may_be_smaller(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=3))
    assert [len(b) for b in batches] == [3, 1]


def test_iter_batches_batch_size_larger_than_store(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == len(docs)


def test_iter_batches_batch_size_one(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=1))
    assert [len(b) for b in batches] == [1, 1, 1, 1]


def test_iter_batches_returns_all_documents(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    result = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert len(result) == len(docs)
    assert {r.id for r in result} == {d.id for d in docs}


def test_iter_batches_matches_all(store: InMemoryDocumentStore, docs: list[Document]) -> None:
    store.add_documents(docs)
    flattened = [doc for batch in store.iter_batches(batch_size=2) for doc in batch]
    assert flattened == store.all()


def test_iter_batches_batches_contain_document_instances(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    batches = list(store.iter_batches(batch_size=2))
    assert all(isinstance(doc, Document) for batch in batches for doc in batch)


def test_iter_batches_zero_batch_size_raises(store: InMemoryDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=0))


def test_iter_batches_negative_batch_size_raises(store: InMemoryDocumentStore) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        list(store.iter_batches(batch_size=-1))


def test_iter_batches_error_raised_before_any_query(store: InMemoryDocumentStore) -> None:
    """The ValueError should be raised eagerly on the first call to
    next(), not silently swallowed by generator laziness."""
    gen = store.iter_batches(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        next(gen)


def test_iter_batches_does_not_mutate_store(
    store: InMemoryDocumentStore, docs: list[Document]
) -> None:
    store.add_documents(docs)
    list(store.iter_batches(batch_size=2))
    assert store.count() == len(docs)


# --- close ---


def test_close_is_no_op(store: InMemoryDocumentStore) -> None:
    store.close()
    store.count()  # still usable after close


def test_close_is_idempotent(store: InMemoryDocumentStore) -> None:
    store.close()
    store.close()  # should not raise


def test_close_returns_none(store: InMemoryDocumentStore) -> None:
    assert store.close() is None


# --- context manager ---


def test_context_manager_returns_self() -> None:
    with InMemoryDocumentStore() as store:
        assert isinstance(store, InMemoryDocumentStore)


def test_context_manager_closes_on_normal_exit() -> None:
    with InMemoryDocumentStore() as store:
        store.add_documents([Document(id="1", page_content="hello", metadata={})])
        assert store.count() == 1

    assert store.count() == 1  # in-memory: still accessible after close


def test_context_manager_closes_on_exception() -> None:
    msg = "boom"
    with pytest.raises(ValueError, match="boom"), InMemoryDocumentStore() as store:
        raise ValueError(msg)

    assert store.count() == 0


def test_context_manager_usable_for_reads_and_writes() -> None:
    with InMemoryDocumentStore() as store:
        store.add_documents(
            [
                Document(id="1", page_content="hello", metadata={"author": "Alice"}),
                Document(id="2", page_content="world", metadata={"author": "Bob"}),
            ]
        )
        assert store.count() == 2
        assert store.filter(author="Alice")[0].id == "1"
        store.delete("1")
        assert store.count() == 1
