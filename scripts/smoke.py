"""End-to-end smoke test against a running server.

Prints the numbers the README quotes, so the README can be checked rather
than trusted. Run the server first, then:

    python scripts/smoke.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8021"
ROOT = Path(__file__).resolve().parent.parent
EMAIL = "demo@tajurba.pk"
PASSWORD = "tajurba-demo-2026"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label} {detail}")


def request(
    path: str, method: str = "GET", body: dict | None = None, token: str | None = None
):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            raw = res.read().decode()
            return res.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def upload(path: str, token: str):
    """Multipart upload without requests, so the script has no dependencies."""
    boundary = "----tajurba-smoke"
    content = (ROOT / path).read_bytes()
    name = Path(path).name
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(BASE + "/api/datasets", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode())


def main() -> int:
    print(f"Smoke test against {BASE}\n")

    print("health")
    status, body = request("/api/health")
    check("GET /api/health is 200", status == 200, f"got {status}")
    check("status is ok", isinstance(body, dict) and body.get("status") == "ok")

    print("\nauth")
    status, body = request(
        "/api/auth/login", "POST", {"email": EMAIL, "password": PASSWORD}
    )
    check("demo login is 200", status == 200, f"got {status}")
    if status != 200:
        print("\nCannot continue without a token.")
        return 1
    token = body["access_token"]

    status, _ = request("/api/datasets")
    check("unauthenticated dataset list is 401", status == 401, f"got {status}")

    print("\nclassification: iris")
    ds = upload("data/iris.csv", token)
    check("iris uploaded", ds["n_rows"] == 150, str(ds))

    status, run = request(
        "/api/runs",
        "POST",
        {"dataset_id": ds["id"], "target": "species", "task": "classification"},
        token,
    )
    check("iris run is 201", status == 201, f"got {status}")
    check("run completed", run["status"] == "completed")
    check(
        "winner beats baseline",
        run["primary_score"] > run["baseline_score"],
        f"{run['primary_score']} vs {run['baseline_score']}",
    )
    check("baseline is 0.3333", abs(run["baseline_score"] - 0.3333) < 0.001)
    check("leaderboard has 6 candidates", len(run["leaderboard"]) == 6)
    check("split adds up", run["train_rows"] + run["test_rows"] == 150)

    print(f"    winner: {run['winner_name']} @ {run['primary_score']}")
    for c in run["leaderboard"]:
        print(f"      {c['name']:42s} {c['scores']}")

    status, pred = request(
        f"/api/runs/{run['id']}/predict",
        "POST",
        {
            "row": {
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "0.2",
            }
        },
        token,
    )
    check("predict is 200", status == 200, f"got {status}")
    check("textbook setosa predicts setosa", pred["prediction"] == "setosa", str(pred))

    print("\nregression: exam scores")
    ds2 = upload("data/exam_scores.csv", token)
    status, run2 = request(
        "/api/runs",
        "POST",
        {"dataset_id": ds2["id"], "target": "final_score", "task": "regression"},
        token,
    )
    check("regression run is 201", status == 201, f"got {status}")
    check("r2 beats baseline", run2["primary_score"] > run2["baseline_score"])
    print(f"    winner: {run2['winner_name']} r2={run2['primary_score']}")
    for c in run2["leaderboard"]:
        print(f"      {c['name']:42s} {c['scores']}")

    print("\nclassification: churn (weak signal)")
    ds3 = upload("data/churn.csv", token)
    status, run3 = request(
        "/api/runs",
        "POST",
        {"dataset_id": ds3["id"], "target": "churned", "task": "classification"},
        token,
    )
    check("churn run is 201", status == 201, f"got {status}")
    print(f"    winner: {run3['winner_name']} acc={run3['primary_score']}")
    print(f"    baseline: {run3['baseline_score']}")
    for c in run3["leaderboard"]:
        print(f"      {c['name']:42s} {c['scores']}")

    print("\nvalidation")
    status, _ = request(
        "/api/runs",
        "POST",
        {"dataset_id": ds["id"], "target": "nope", "task": "classification"},
        token,
    )
    check("unknown target is 422", status == 422, f"got {status}")

    status, _ = request("/api/datasets/doesnotexist", token=token)
    check("unknown dataset is 404", status == 404, f"got {status}")

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
