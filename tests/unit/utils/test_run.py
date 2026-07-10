from __future__ import annotations

import uuid

import pytest

from zenpyre.utils.run import extract_run_id, extract_run_ids, generate_run_id

###################################
#     Tests for extract_run_id    #
###################################


def test_extract_run_id_none_config() -> None:
    assert extract_run_id(None) is None


def test_extract_run_id_with_run_id() -> None:
    assert extract_run_id({"run_id": "abc-123"}) == "abc-123"


def test_extract_run_id_missing_run_id() -> None:
    assert extract_run_id({"tags": ["a", "b"]}) is None


def test_extract_run_id_empty_config() -> None:
    assert extract_run_id({}) is None


def test_extract_run_id_run_id_is_none() -> None:
    assert extract_run_id({"run_id": None}) is None


def test_extract_run_id_non_string_run_id_converted_to_str() -> None:
    # run_id may be a UUID object rather than a plain string.
    import uuid

    run_id = uuid.uuid4()
    assert extract_run_id({"run_id": run_id}) == str(run_id)


def test_extract_run_id_config_with_extra_keys() -> None:
    config = {"run_id": "abc-123", "tags": ["x"], "metadata": {"key": "value"}}
    assert extract_run_id(config) == "abc-123"


####################################
#     Tests for extract_run_ids    #
####################################


def test_extract_run_ids_none_config() -> None:
    assert extract_run_ids(None, n=3) == [None, None, None]


def test_extract_run_ids_none_config_zero_items() -> None:
    assert extract_run_ids(None, n=0) == []


def test_extract_run_ids_single_shared_config() -> None:
    assert extract_run_ids({"run_id": "abc"}, n=3) == ["abc", "abc", "abc"]


def test_extract_run_ids_single_shared_config_without_run_id() -> None:
    assert extract_run_ids({"tags": ["a"]}, n=2) == [None, None]


def test_extract_run_ids_list_of_configs() -> None:
    configs = [{"run_id": "a"}, {"run_id": "b"}]
    assert extract_run_ids(configs, n=2) == ["a", "b"]


def test_extract_run_ids_list_of_configs_mixed_missing_run_id() -> None:
    configs = [{"run_id": "a"}, {"tags": ["x"]}, {"run_id": "c"}]
    assert extract_run_ids(configs, n=3) == ["a", None, "c"]


def test_extract_run_ids_empty_list_matching_n() -> None:
    assert extract_run_ids([], n=0) == []


def test_extract_run_ids_list_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Expected 3 configs but received 2"):
        extract_run_ids([{"run_id": "a"}, {"run_id": "b"}], n=3)


def test_extract_run_ids_list_length_mismatch_empty_list() -> None:
    with pytest.raises(ValueError, match="Expected 2 configs but received 0"):
        extract_run_ids([], n=2)


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_extract_run_ids_single_shared_config_various_n(n: int) -> None:
    result = extract_run_ids({"run_id": "abc"}, n=n)
    assert result == ["abc"] * n
    assert len(result) == n


#####################################
#     Tests for generate_run_id     #
#####################################


def test_generate_run_id_no_config_returns_str() -> None:
    assert isinstance(generate_run_id(), str)


def test_generate_run_id_no_config_is_valid_uuid() -> None:
    # generate_run_id() should return a string parseable as a UUID.
    assert uuid.UUID(generate_run_id()) is not None


def test_generate_run_id_no_config_is_random() -> None:
    assert generate_run_id() != generate_run_id()


def test_generate_run_id_none_config_is_random() -> None:
    assert generate_run_id(None) != generate_run_id(None)


def test_generate_run_id_with_config_returns_str() -> None:
    assert isinstance(generate_run_id({"lr": 0.01, "batch_size": 32}), str)


def test_generate_run_id_same_config_gives_same_id() -> None:
    config = {"lr": 0.01, "batch_size": 32}
    assert generate_run_id(config) == generate_run_id(config)


def test_generate_run_id_equal_but_distinct_configs_give_same_id() -> None:
    # Two distinct dict objects with equal content should produce the same ID.
    config1 = {"lr": 0.01, "batch_size": 32}
    config2 = {"lr": 0.01, "batch_size": 32}
    assert generate_run_id(config1) == generate_run_id(config2)


def test_generate_run_id_different_configs_give_different_ids() -> None:
    assert generate_run_id({"lr": 0.01}) != generate_run_id({"lr": 0.02})


def test_generate_run_id_different_keys_give_different_ids() -> None:
    assert generate_run_id({"a": 1}) != generate_run_id({"b": 1})


def test_generate_run_id_empty_config_returns_str() -> None:
    assert isinstance(generate_run_id({}), str)


def test_generate_run_id_none_and_empty_config_differ() -> None:
    assert generate_run_id(None) != generate_run_id({})
