from __future__ import annotations

from langchain_core.documents import Document

from zenpyre.documents import format_documents_as_xml

######################################################
#     Tests for format_documents_as_xml              #
######################################################


def test_format_documents_as_xml_empty() -> None:
    assert format_documents_as_xml([]) == ""


def test_format_documents_as_xml_single_document() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    assert format_documents_as_xml(documents) == (
        '<document id="1">\nThe cat sat on the mat.\n</document>'
    )


def test_format_documents_as_xml_multiple_documents() -> None:
    documents = [
        Document(page_content="The cat sat on the mat."),
        Document(page_content="The dog chased the ball."),
    ]
    assert format_documents_as_xml(documents) == (
        '<document id="1">\n'
        "The cat sat on the mat.\n"
        "</document>\n"
        "\n"
        '<document id="2">\n'
        "The dog chased the ball.\n"
        "</document>"
    )


def test_format_documents_as_xml_without_metadata_ignores_metadata() -> None:
    documents = [
        Document(page_content="The cat sat on the mat.", metadata={"source": "story.txt"}),
    ]
    assert format_documents_as_xml(documents, include_metadata=False) == (
        '<document id="1">\nThe cat sat on the mat.\n</document>'
    )


def test_format_documents_as_xml_with_metadata_single_document() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"source": "story.txt", "author": "Alice"},
        ),
    ]
    assert format_documents_as_xml(documents, include_metadata=True) == (
        '<document id="1">\n'
        "author: Alice\n"
        "source: story.txt\n"
        "\n"
        "The cat sat on the mat.\n"
        "</document>"
    )


def test_format_documents_as_xml_with_metadata_multiple_documents() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"source": "story.txt", "author": "Alice"},
        ),
        Document(
            page_content="The dog chased the ball.",
            metadata={"source": "story.txt", "author": "Bob"},
        ),
    ]
    assert format_documents_as_xml(documents, include_metadata=True) == (
        '<document id="1">\n'
        "author: Alice\n"
        "source: story.txt\n"
        "\n"
        "The cat sat on the mat.\n"
        "</document>\n"
        "\n"
        '<document id="2">\n'
        "author: Bob\n"
        "source: story.txt\n"
        "\n"
        "The dog chased the ball.\n"
        "</document>"
    )


def test_format_documents_as_xml_with_metadata_sorted_by_key() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"zebra": "z", "apple": "a", "mango": "m"},
        ),
    ]
    assert format_documents_as_xml(documents, include_metadata=True) == (
        '<document id="1">\napple: a\nmango: m\nzebra: z\n\nThe cat sat on the mat.\n</document>'
    )


def test_format_documents_as_xml_with_metadata_true_but_empty_metadata() -> None:
    documents = [Document(page_content="The cat sat on the mat.", metadata={})]
    assert format_documents_as_xml(documents, include_metadata=True) == (
        '<document id="1">\nThe cat sat on the mat.\n</document>'
    )
