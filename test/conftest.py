# SPDX-License-Identifier: MPL-2.0
from ragql.config import Settings
import numpy as np
import pytest
import warnings

warnings.filterwarnings("ignore", message="numpy.core._multiarray_umath is deprecated")

@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Settings that isolate each test run in a temp DB."""
    db_path = tmp_path_factory.mktemp("dbs") / "ragql.db"
    return Settings(db_path=db_path, use_ollama=False, openai_key="DUMMY")

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Disable HTTP during tests so accidental calls fail fast."""
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: None)
