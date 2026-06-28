from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _session_env() -> None:
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-fake")
    os.environ.setdefault("OPENAI_API_KEY", "sk-fake")
    os.environ.setdefault("SEC_EMAIL", "t7xk2m.w9qsec4r8@gmail.com")
