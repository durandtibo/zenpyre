from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.panel import Panel

from zenpyre.utils.structured import (
    format_structured_output_schema,
    get_structured_output_tool_spec,
    print_structured_output_schema,
)

MODULE = "zenpyre.utils.structured"


# ---------------------------------------------------------------------------
# Minimal schema fixtures
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    """Extract a single entity mentioned in the user's message."""

    name: str = Field(description="The full name of the entity, as written.")
    category: str = Field(description="One of 'person', 'place', or 'organization'.")
    confidence: float = Field(default=1.0, description="How confident you are, from 0 to 1.")


class Undocumented(BaseModel):
    name: str
    age: int = 0


####################################################
#     Tests for get_structured_output_tool_spec    #
####################################################


def test_get_structured_output_tool_spec_name() -> None:
    spec = get_structured_output_tool_spec(Entity)
    assert spec["function"]["name"] == "Entity"


def test_get_structured_output_tool_spec_docstring_as_description() -> None:
    spec = get_structured_output_tool_spec(Entity)
    assert (
        spec["function"]["description"]
        == "Extract a single entity mentioned in the user's message."
    )


def test_get_structured_output_tool_spec_field_descriptions() -> None:
    spec = get_structured_output_tool_spec(Entity)
    properties = spec["function"]["parameters"]["properties"]
    assert properties["name"]["description"] == "The full name of the entity, as written."
    assert properties["category"]["description"] == "One of 'person', 'place', or 'organization'."


def test_get_structured_output_tool_spec_required_fields() -> None:
    spec = get_structured_output_tool_spec(Entity)
    assert spec["function"]["parameters"]["required"] == ["name", "category"]


def test_get_structured_output_tool_spec_no_docstring() -> None:
    spec = get_structured_output_tool_spec(Undocumented)
    assert not spec["function"].get("description")


###################################################
#     Tests for format_structured_output_schema  #
###################################################


def test_format_structured_output_schema_contains_class_name() -> None:
    assert "Class: Entity" in format_structured_output_schema(Entity)


def test_format_structured_output_schema_contains_docstring() -> None:
    result = format_structured_output_schema(Entity)
    assert "Extract a single entity mentioned in the user's message." in result


def test_format_structured_output_schema_contains_json_schema() -> None:
    result = format_structured_output_schema(Entity)
    assert '"type": "object"' in result
    assert '"required"' in result


def test_format_structured_output_schema_contains_field_breakdown() -> None:
    result = format_structured_output_schema(Entity)
    assert "- name (string, required): 'The full name of the entity, as written.'" in result
    assert "- confidence (number, optional):" in result


def test_format_structured_output_schema_missing_description_flagged() -> None:
    result = format_structured_output_schema(Undocumented)
    assert "<no description set>" in result


def test_format_structured_output_schema_returns_str() -> None:
    assert isinstance(format_structured_output_schema(Entity), str)


###################################################
#     Tests for print_structured_output_schema   #
###################################################


def test_print_structured_output_schema_returns_none() -> None:
    assert print_structured_output_schema(Entity) is None


def test_print_structured_output_schema_renders_panel() -> None:
    with patch(f"{MODULE}.get_console") as mock_get_console:
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console
        print_structured_output_schema(Entity)
    panel: Panel = mock_console.print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)


def test_print_structured_output_schema_title_contains_class_name() -> None:
    custom = MagicMock(spec=Console)
    print_structured_output_schema(Entity, console=custom)
    panel: Panel = custom.print.call_args.args[0]
    assert panel.title is not None
    assert "Entity" in panel.title


def test_print_structured_output_schema_uses_custom_console() -> None:
    custom = MagicMock(spec=Console)
    print_structured_output_schema(Entity, console=custom)
    custom.print.assert_called_once()


def test_print_structured_output_schema_custom_console_not_shared() -> None:
    """A per-call console does not affect the shared instance."""
    custom = MagicMock(spec=Console)
    with patch(f"{MODULE}.get_console") as mock_get_console:
        mock_get_console.return_value = MagicMock(spec=Console)
        print_structured_output_schema(Entity, console=custom)
        assert mock_get_console.return_value is not custom


@pytest.mark.parametrize("schema", [Entity, Undocumented])
def test_print_structured_output_schema_renders_for_any_schema(schema: type[BaseModel]) -> None:
    custom = MagicMock(spec=Console)
    print_structured_output_schema(schema, console=custom)
    custom.print.assert_called_once()
