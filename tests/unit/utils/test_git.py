# ruff: noqa: S607 -- all subprocess calls in this module are fixed,
# hardcoded `git` invocations used to build a throwaway repository fixture;
# there is no user-controlled input.
from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from zenpyre.utils.git import get_current_commit_hash

if TYPE_CHECKING:
    from pathlib import Path


MODULE = "zenpyre.utils.git"


def _init_repo(path: Path) -> None:
    """Initialize a git repository with a single commit at ``path``."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


##############################################
#     Tests for get_current_commit_hash      #
##############################################


# --- Real repository ---


def test_get_current_commit_hash_returns_full_hash(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    commit_hash = get_current_commit_hash(cwd=tmp_path)
    assert commit_hash is not None
    assert len(commit_hash) == 40


def test_get_current_commit_hash_returns_short_hash(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    commit_hash = get_current_commit_hash(short=True, cwd=tmp_path)
    assert commit_hash is not None
    assert 4 <= len(commit_hash) < 40


def test_get_current_commit_hash_short_is_prefix_of_full(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    full = get_current_commit_hash(cwd=tmp_path)
    short = get_current_commit_hash(short=True, cwd=tmp_path)
    assert full.startswith(short)


def test_get_current_commit_hash_default_short_is_false(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert get_current_commit_hash(cwd=tmp_path) == get_current_commit_hash(
        short=False, cwd=tmp_path
    )


# --- Not a git repository ---


def test_get_current_commit_hash_outside_repo_returns_none(tmp_path: Path) -> None:
    assert get_current_commit_hash(cwd=tmp_path) is None


def test_get_current_commit_hash_empty_repo_returns_none(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    assert get_current_commit_hash(cwd=tmp_path) is None


# --- git executable unavailable ---


def test_get_current_commit_hash_missing_git_executable_returns_none(tmp_path: Path) -> None:
    with patch(f"{MODULE}.subprocess.check_output", side_effect=FileNotFoundError):
        assert get_current_commit_hash(cwd=tmp_path) is None


def test_get_current_commit_hash_os_error_returns_none(tmp_path: Path) -> None:
    with patch("zenpyre.utils.git.subprocess.check_output", side_effect=OSError):
        assert get_current_commit_hash(cwd=tmp_path) is None


# --- default cwd ---


def test_get_current_commit_hash_uses_current_process_cwd_by_default() -> None:
    with patch(f"{MODULE}.subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = "abc1234\n"
        get_current_commit_hash()
        assert mock_check_output.call_args.kwargs["cwd"] is None
