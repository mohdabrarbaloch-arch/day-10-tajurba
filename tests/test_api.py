"""End-to-end API tests through the real FastAPI app.

The integration test walks the whole product path — register, upload,
train, predict — because that is the sequence a user actually performs and
the only way to catch a break between layers.
"""

from __future__ import annotations


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_register_returns_token(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "a@example.com", "name": "A", "password": "good-password-1"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["access_token"]
    assert body["user"]["email"] == "a@example.com"


def test_register_rejects_duplicate_email(client):
    payload = {"email": "dup@example.com", "name": "D", "password": "good-password-1"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_register_rejects_short_password(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "b@example.com", "name": "B", "password": "short"},
    )
    assert res.status_code == 422


def test_login_wrong_password_is_401(client):
    client.post(
        "/api/auth/register",
        json={"email": "c@example.com", "name": "C", "password": "good-password-1"},
    )
    res = client.post(
        "/api/auth/login", json={"email": "c@example.com", "password": "wrong-one-9"}
    )
    assert res.status_code == 401


def test_protected_route_requires_token(client):
    assert client.get("/api/datasets").status_code == 401


def test_me_returns_current_user(auth_client):
    res = auth_client.get("/api/auth/me")
    assert res.status_code == 200
    assert res.json()["name"] == "Test User"


def test_upload_rejects_non_csv(auth_client):
    res = auth_client.post(
        "/api/datasets",
        files={"file": ("data.txt", b"a,b\n1,2\n", "text/plain")},
    )
    assert res.status_code == 415


def test_upload_rejects_tiny_file(auth_client):
    res = auth_client.post(
        "/api/datasets",
        files={"file": ("tiny.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
    )
    assert res.status_code == 422


def test_upload_and_inspect_dataset(auth_client, iris_csv):
    res = auth_client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
    )
    assert res.status_code == 201
    ds = res.json()
    assert ds["n_rows"] == 150

    detail = auth_client.get(f"/api/datasets/{ds['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert "species" in body["columns"]
    assert len(body["preview"]) > 0
    assert body["suggested_target"] == "species"


def test_dataset_isolation_between_users(client, iris_csv):
    """One user must not be able to read another user's dataset."""
    first = client.post(
        "/api/auth/register",
        json={"email": "u1@example.com", "name": "U1", "password": "good-password-1"},
    ).json()["access_token"]

    res = client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
        headers={"Authorization": f"Bearer {first}"},
    )
    ds_id = res.json()["id"]

    second = client.post(
        "/api/auth/register",
        json={"email": "u2@example.com", "name": "U2", "password": "good-password-1"},
    ).json()["access_token"]

    res = client.get(
        f"/api/datasets/{ds_id}", headers={"Authorization": f"Bearer {second}"}
    )
    assert res.status_code == 404


def test_full_flow_upload_train_predict(auth_client, iris_csv):
    """The whole product path, end to end."""
    up = auth_client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
    )
    assert up.status_code == 201
    ds_id = up.json()["id"]

    run = auth_client.post(
        "/api/runs",
        json={"dataset_id": ds_id, "target": "species", "task": "classification"},
    )
    assert run.status_code == 201, run.text
    body = run.json()

    assert body["status"] == "completed"
    assert body["winner_name"]
    assert body["primary_score"] > body["baseline_score"]
    assert len(body["leaderboard"]) >= 6
    assert body["train_rows"] + body["test_rows"] == 150

    pred = auth_client.post(
        f"/api/runs/{body['id']}/predict",
        json={
            "row": {
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "0.2",
            }
        },
    )
    assert pred.status_code == 200, pred.text
    result = pred.json()
    assert result["prediction"] in {"setosa", "versicolor", "virginica"}
    # A textbook setosa measurement should land on setosa.
    assert result["prediction"] == "setosa"


def test_train_rejects_unknown_dataset(auth_client):
    res = auth_client.post(
        "/api/runs",
        json={"dataset_id": "doesnotexist", "target": "x", "task": "classification"},
    )
    assert res.status_code == 404


def test_train_rejects_missing_target(auth_client, iris_csv):
    up = auth_client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
    )
    ds_id = up.json()["id"]
    res = auth_client.post(
        "/api/runs",
        json={"dataset_id": ds_id, "target": "nope", "task": "classification"},
    )
    assert res.status_code == 422


def test_runs_are_listed_newest_first(auth_client, iris_csv):
    up = auth_client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
    )
    ds_id = up.json()["id"]
    auth_client.post(
        "/api/runs",
        json={"dataset_id": ds_id, "target": "species", "task": "classification"},
    )
    res = auth_client.get("/api/runs")
    assert res.status_code == 200
    assert len(res.json()) >= 1


def test_delete_dataset_returns_204(auth_client, iris_csv):
    up = auth_client.post(
        "/api/datasets",
        files={"file": ("iris.csv", iris_csv.encode(), "text/csv")},
    )
    ds_id = up.json()["id"]
    assert auth_client.delete(f"/api/datasets/{ds_id}").status_code == 204
    assert auth_client.get(f"/api/datasets/{ds_id}").status_code == 404


def test_samples_endpoint_lists_three(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    assert len(res.json()) == 3


def test_sample_download_returns_csv(client):
    res = client.get("/api/samples/iris")
    assert res.status_code == 200
    assert "species" in res.text


def test_security_headers_present(client):
    res = client.get("/api/health")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
