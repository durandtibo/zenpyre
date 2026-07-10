from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from zenpyre.utils.serialization import default_serialize

######################################
#     Tests for default_serialize    #
######################################


class Answer(BaseModel):
    value: int


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Line:
    start: Point
    end: Point


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc", id="str"),
        pytest.param(42, id="int"),
        pytest.param(3.14, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
    ],
)
def test_default_serialize_plain_types_unchanged(value: object) -> None:
    assert default_serialize(value) == value


def test_default_serialize_pydantic_model() -> None:
    assert default_serialize(Answer(value=5)) == {"value": 5}


def test_default_serialize_dict() -> None:
    assert default_serialize({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_default_serialize_dict_with_pydantic_values() -> None:
    assert default_serialize({"result": Answer(value=5)}) == {"result": {"value": 5}}


def test_default_serialize_list() -> None:
    assert default_serialize([1, "a", 2]) == [1, "a", 2]


def test_default_serialize_list_with_pydantic_values() -> None:
    assert default_serialize([Answer(value=1), "x"]) == [{"value": 1}, "x"]


def test_default_serialize_tuple_converted_to_list() -> None:
    assert default_serialize((1, 2, 3)) == [1, 2, 3]


def test_default_serialize_nested_structure() -> None:
    result = default_serialize({"result": Answer(value=5), "items": [Answer(value=1), "x"]})
    assert result == {"result": {"value": 5}, "items": [{"value": 1}, "x"]}


def test_default_serialize_dataclass_instance() -> None:
    assert default_serialize(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_default_serialize_dataclass_class_unchanged() -> None:
    # The class itself (not an instance) should be returned as-is,
    # not passed to dataclasses.asdict (which requires an instance).
    assert default_serialize(Point) is Point


def test_default_serialize_dict_with_dataclass_values() -> None:
    assert default_serialize({"point": Point(x=1, y=2)}) == {"point": {"x": 1, "y": 2}}


def test_default_serialize_list_with_dataclass_values() -> None:
    assert default_serialize([Point(x=1, y=2), "x"]) == [{"x": 1, "y": 2}, "x"]


def test_default_serialize_dataclass_with_pydantic_field() -> None:
    @dataclass
    class Wrapper:
        answer: Answer

    assert default_serialize(Wrapper(answer=Answer(value=5))) == {"answer": {"value": 5}}


def test_default_serialize_dataclass_nested_dataclass() -> None:
    # Known limitation: dataclasses.asdict recurses into nested
    # dataclasses on its own, before default_serialize gets a chance
    # to route them through model_dump-aware logic. For pure nested
    # dataclasses (no Pydantic model buried inside), the plain-dict
    # result is still correct.
    line = Line(start=Point(x=0, y=0), end=Point(x=1, y=1))
    assert default_serialize(line) == {"start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}


def test_default_serialize_dataclass_with_nested_dataclass_containing_pydantic() -> None:
    # dataclasses.asdict recurses into nested dataclass fields (Inner),
    # but leaves non-dataclass field values (the Pydantic Answer) as-is
    # inside the resulting dict. default_serialize's own dict-recursion
    # then picks up that leftover Answer instance and correctly routes
    # it through model_dump. So this nested case works correctly.
    @dataclass
    class Inner:
        answer: Answer

    @dataclass
    class Outer:
        inner: Inner

    result = default_serialize(Outer(inner=Inner(answer=Answer(value=5))))
    assert result == {"inner": {"answer": {"value": 5}}}


def test_default_serialize_dataclass_with_list_field() -> None:
    @dataclass
    class Group:
        points: list

    result = default_serialize(Group(points=[Point(x=1, y=2), Point(x=3, y=4)]))
    assert result == {"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
