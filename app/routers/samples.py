"""The three bundled datasets, served so a new user can try the app in one click.

These are the same files `scripts/make_samples.py` writes. Serving them from
the API rather than shipping them in the frontend means the "load sample"
button exercises the exact same upload path a real file takes.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api/samples", tags=["samples"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

SAMPLES = {
    "iris": {
        "file": "iris.csv",
        "title": "Iris species",
        "task": "classification",
        "target": "species",
        "blurb": "150 flowers, 3 species, 4 measurements. The hello-world of classification.",
    },
    "churn": {
        "file": "churn.csv",
        "title": "Customer churn",
        "task": "classification",
        "target": "churned",
        "blurb": "400 subscribers with a weak, realistic signal. Watch the baseline matter here.",
    },
    "exam": {
        "file": "exam_scores.csv",
        "title": "Exam scores",
        "task": "regression",
        "target": "final_score",
        "blurb": "300 students, mostly linear with a little curvature. Ridge should win.",
    },
}


@router.get("")
def list_samples() -> list[dict]:
    out = []
    for key, meta in SAMPLES.items():
        path = DATA_DIR / meta["file"]
        out.append(
            {
                "key": key,
                "title": meta["title"],
                "task": meta["task"],
                "target": meta["target"],
                "blurb": meta["blurb"],
                "available": path.exists(),
            }
        )
    return out


@router.get("/{key}", response_class=PlainTextResponse)
def get_sample(key: str) -> PlainTextResponse:
    meta = SAMPLES.get(key)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No such sample."
        )
    path = DATA_DIR / meta["file"]
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sample data is missing. Run scripts/make_samples.py.",
        )
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": f'attachment; filename="{meta["file"]}"'},
    )
