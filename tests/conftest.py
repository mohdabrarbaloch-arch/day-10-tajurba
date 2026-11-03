"""Test fixtures.

Each test gets its own SQLite file in a temp directory, so a failing test
can never leave state behind for the next one.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before app.config is imported anywhere.
_tmpdir = tempfile.mkdtemp(prefix="tajurba-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/test.db"
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"


@pytest.fixture(scope="session")
def app_module():
    from app.main import app

    return app


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient

    from app import ratelimit

    # Each test starts with a clean rate-limit window, so a test that
    # registers several users cannot trip the limiter for the next one.
    ratelimit.reset()
    with TestClient(app_module) as c:
        yield c


@pytest.fixture()
def auth_client(client):
    """A client already signed in as a fresh user."""
    import uuid

    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    res = client.post(
        "/api/auth/register",
        json={"email": email, "name": "Test User", "password": "correct-horse-9"},
    )
    assert res.status_code == 201, res.text
    token = res.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture()
def iris_csv() -> str:
    return (ROOT / "data" / "iris.csv").read_text(encoding="utf-8")


@pytest.fixture()
def churn_csv() -> str:
    return (ROOT / "data" / "churn.csv").read_text(encoding="utf-8")


@pytest.fixture()
def exam_csv() -> str:
    return (ROOT / "data" / "exam_scores.csv").read_text(encoding="utf-8")
