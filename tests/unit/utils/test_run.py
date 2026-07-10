from __future__ import annotations

import uuid

from zenpyre.utils.run import generate_run_id

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
