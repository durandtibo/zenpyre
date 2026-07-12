from __future__ import annotations

import pytest
from langchain_core.documents import Document

from zenpyre.documents import (
    format_documents,
    format_documents_as_markdown,
    format_documents_as_xml,
)

######################################
#     Tests for format_documents     #
######################################


def test_format_documents_default_is_xml() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    assert format_documents(documents) == format_documents_as_xml(documents)


def test_format_documents_output_format_xml() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    assert format_documents(documents, output_format="xml") == format_documents_as_xml(documents)


def test_format_documents_output_format_markdown() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    assert format_documents(documents, output_format="markdown") == format_documents_as_markdown(
        documents
    )


def test_format_documents_output_format_markdown_with_metadata() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"source": "story.txt", "author": "Alice"},
        ),
    ]
    assert format_documents(
        documents, include_metadata=True, output_format="markdown"
    ) == format_documents_as_markdown(documents, include_metadata=True)


def test_format_documents_output_format_xml_with_metadata() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"source": "story.txt", "author": "Alice"},
        ),
    ]
    assert format_documents(
        documents, include_metadata=True, output_format="xml"
    ) == format_documents_as_xml(documents, include_metadata=True)


def test_format_documents_empty() -> None:
    assert format_documents([]) == ""
    assert format_documents([], output_format="markdown") == ""


def test_format_documents_invalid_output_format_raises() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    with pytest.raises(ValueError, match="Unknown format: bogus"):
        format_documents(documents, output_format="bogus")


######################################################
#     Tests for format_documents_as_markdown         #
######################################################


def test_format_documents_as_markdown_empty() -> None:
    assert format_documents_as_markdown([]) == ""


def test_format_documents_as_markdown_single_document() -> None:
    documents = [Document(page_content="The cat sat on the mat.")]
    assert format_documents_as_markdown(documents) == ("## Document 1\n\nThe cat sat on the mat.")


def test_format_documents_as_markdown_multiple_documents() -> None:
    documents = [
        Document(page_content="The cat sat on the mat."),
        Document(page_content="The dog chased the ball."),
    ]
    assert format_documents_as_markdown(documents) == (
        "## Document 1\n\nThe cat sat on the mat.\n\n## Document 2\n\nThe dog chased the ball."
    )


def test_format_documents_as_markdown_without_metadata_ignores_metadata() -> None:
    documents = [
        Document(page_content="The cat sat on the mat.", metadata={"source": "story.txt"}),
    ]
    assert format_documents_as_markdown(documents, include_metadata=False) == (
        "## Document 1\n\nThe cat sat on the mat."
    )


def test_format_documents_as_markdown_with_metadata_single_document() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"source": "story.txt", "author": "Alice"},
        ),
    ]
    assert format_documents_as_markdown(documents, include_metadata=True) == (
        "## Document 1\n\n- author: Alice\n- source: story.txt\n\nThe cat sat on the mat."
    )


def test_format_documents_as_markdown_with_metadata_multiple_documents() -> None:
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
    assert format_documents_as_markdown(documents, include_metadata=True) == (
        "## Document 1\n"
        "\n"
        "- author: Alice\n"
        "- source: story.txt\n"
        "\n"
        "The cat sat on the mat.\n"
        "\n"
        "## Document 2\n"
        "\n"
        "- author: Bob\n"
        "- source: story.txt\n"
        "\n"
        "The dog chased the ball."
    )


def test_format_documents_as_markdown_with_metadata_sorted_by_key() -> None:
    documents = [
        Document(
            page_content="The cat sat on the mat.",
            metadata={"zebra": "z", "apple": "a", "mango": "m"},
        ),
    ]
    assert format_documents_as_markdown(documents, include_metadata=True) == (
        "## Document 1\n\n- apple: a\n- mango: m\n- zebra: z\n\nThe cat sat on the mat."
    )


def test_format_documents_as_markdown_with_metadata_true_but_empty_metadata() -> None:
    documents = [Document(page_content="The cat sat on the mat.", metadata={})]
    assert format_documents_as_markdown(documents, include_metadata=True) == (
        "## Document 1\n\nThe cat sat on the mat."
    )


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
