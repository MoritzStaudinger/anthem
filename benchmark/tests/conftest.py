import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def flask_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "temp").mkdir(parents=True, exist_ok=True)

    if "app" in sys.modules:
        del sys.modules["app"]
    if "backend.api.routes" in sys.modules:
        del sys.modules["backend.api.routes"]

    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c
