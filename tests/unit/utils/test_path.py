from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zenpyre.utils.path import find_root_package_parent

if TYPE_CHECKING:
    import pytest

##############################################
#     Tests for find_root_package_parent     #
##############################################


# --- Nested package resolution ---


def test_find_root_package_parent_from_file_in_nested_package(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (sub / "__init__.py").touch()
    module = sub / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


def test_find_root_package_parent_from_directory_in_nested_package(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (sub / "__init__.py").touch()

    assert find_root_package_parent(sub) == tmp_path


def test_find_root_package_parent_from_top_level_package_file(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


# --- Non-package input ---


def test_find_root_package_parent_file_without_init_returns_own_dir(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


def test_find_root_package_parent_directory_without_init_returns_itself(tmp_path: Path) -> None:
    directory = tmp_path / "not_a_package"
    directory.mkdir()

    assert find_root_package_parent(directory) == directory


# --- String input ---


def test_find_root_package_parent_accepts_string_path(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    assert find_root_package_parent(str(module)) == tmp_path


# --- Return type ---


def test_find_root_package_parent_returns_path_instance(tmp_path: Path) -> None:
    directory = tmp_path / "not_a_package"
    directory.mkdir()

    assert isinstance(find_root_package_parent(directory), Path)


# --- Relative path resolution ---


def test_find_root_package_parent_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    monkeypatch.chdir(tmp_path)
    assert find_root_package_parent(Path("pkg") / "module.py") == tmp_path
