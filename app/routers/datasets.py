"""Dataset upload, listing and inspection."""

from __future__ import annotations

import json

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..automl.csv_io import CsvError, parse_csv
from ..automl.dataset import detect_task
from ..config import get_settings
from ..db import get_db
from ..models import Dataset, User
from ..schemas import DatasetDetail, DatasetOut
from ..security import current_user

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
settings = get_settings()

PREVIEW_ROWS = 8

# Column names that almost always hold the thing you want to predict.
TARGET_HINTS = (
    "target",
    "label",
    "class",
    "species",
    "churned",
    "outcome",
    "result",
    "score",
    "price",
    "sales",
    "revenue",
    "y",
    "survived",
    "default",
)


def suggest_target(table, columns: list[str]) -> str | None:
    """Guess which column is the label.

    Three signals, in order of confidence:
      1. A name that matches a known target convention.
      2. A low-cardinality column (2-20 distinct values) — that is what a
         label looks like, and it is why "species" beats "petal_width".
      3. Fall back to the last column, the usual export convention.
    """
    if not columns:
        return None

    # Substring, not exact match: "final_score" and "churn_label" should both
    # hit. Checked longest-hint-first so "score" does not shadow "prior_score"
    # style names by accident.
    for col in columns:
        lowered = col.strip().lower()
        if any(hint in lowered for hint in TARGET_HINTS):
            return col

    best: tuple[float, str] | None = None
    for col in columns:
        values = [row[col] for row in table.rows if row[col] != ""]
        if not values:
            continue
        distinct = len(set(values))
        ratio = distinct / len(values)
        # A label is a small set of repeated values, not a near-unique id.
        if 2 <= distinct <= 20 and ratio < 0.5:
            score = ratio  # lower ratio = more label-like
            if best is None or score < best[0]:
                best = (score, col)

    if best is not None:
        return best[1]

    return columns[-1]


def _owned(db: Session, dataset_id: str, user: User) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None or ds.owner_id != user.id:
        # 404 rather than 403: telling someone a dataset exists but isn't
        # theirs is itself a leak.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found."
        )
    return ds


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DatasetOut:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv files are accepted.",
        )

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {len(raw) / 1024 / 1024:.1f} MB; the limit is "
                f"{settings.max_upload_bytes / 1024 / 1024:.0f} MB."
            ),
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The file is not valid UTF-8 or Latin-1 text.",
            ) from exc

    try:
        table = parse_csv(text, max_rows=settings.max_rows)
    except CsvError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    ds = Dataset(
        owner_id=user.id,
        name=file.filename.rsplit(".", 1)[0][:200],
        filename=file.filename[:255],
        content=text,
        n_rows=len(table.rows),
        n_columns=len(table.columns),
        columns_json=json.dumps(table.columns),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return DatasetOut.model_validate(ds)


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[DatasetOut]:
    rows = db.scalars(
        select(Dataset)
        .where(Dataset.owner_id == user.id)
        .order_by(Dataset.created_at.desc())
    ).all()
    return [DatasetOut.model_validate(r) for r in rows]


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DatasetDetail:
    ds = _owned(db, dataset_id, user)
    table = parse_csv(ds.content, max_rows=settings.max_rows)
    columns = json.loads(ds.columns_json)

    suggested = suggest_target(table, columns)
    suggested_task = detect_task(table, suggested) if suggested else None

    return DatasetDetail(
        id=ds.id,
        name=ds.name,
        filename=ds.filename,
        n_rows=ds.n_rows,
        n_columns=ds.n_columns,
        created_at=ds.created_at,
        columns=columns,
        preview=table.rows[:PREVIEW_ROWS],
        suggested_target=suggested,
        suggested_task=suggested_task,
    )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Response:
    ds = _owned(db, dataset_id, user)
    db.delete(ds)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
