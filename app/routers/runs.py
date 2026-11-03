"""Training runs, the leaderboard, and scoring new rows."""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..automl import metrics
from ..automl.csv_io import ParsedTable, parse_csv
from ..automl.dataset import DatasetError, FeatureSpec, transform
from ..automl.trainer import train
from ..config import get_settings
from ..db import get_db
from ..models import Dataset, Run, User
from ..schemas import (
    CandidateOut,
    PredictRequest,
    PredictResponse,
    RunDetail,
    RunOut,
    TrainRequest,
)
from ..security import current_user

router = APIRouter(prefix="/api/runs", tags=["runs"])
settings = get_settings()


def _owned_run(db: Session, run_id: str, user: User) -> Run:
    run = db.get(Run, run_id)
    if run is None or run.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Run not found."
        )
    return run


@router.post("", response_model=RunDetail, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: TrainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> RunDetail:
    ds = db.get(Dataset, payload.dataset_id)
    if ds is None or ds.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found."
        )

    try:
        table = parse_csv(ds.content, max_rows=settings.max_rows)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Stored dataset could not be re-parsed: {exc}",
        ) from exc

    started = time.perf_counter()
    try:
        report = train(
            table,
            target=payload.target,
            task=payload.task,
            test_fraction=payload.test_fraction,
            seed=payload.seed,
            feature_columns=payload.feature_columns,
        )
    except DatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    duration = time.perf_counter() - started

    primary = metrics.primary_metric_name(payload.task)
    winner = report.winner

    run = Run(
        owner_id=user.id,
        dataset_id=ds.id,
        target=payload.target,
        task=payload.task,
        status="completed" if winner else "failed",
        winner_name=winner.name if winner else None,
        winner_family=winner.family if winner else None,
        primary_metric=primary,
        primary_score=winner.scores.get(primary) if winner else None,
        baseline_score=report.baseline.get(primary),
        leaderboard_json=json.dumps(
            [
                {
                    "name": c.name,
                    "family": c.family,
                    "params": c.params,
                    "scores": c.scores,
                    "train_seconds": c.train_seconds,
                    "error": c.error,
                }
                for c in report.leaderboard
            ]
        ),
        baseline_json=json.dumps(report.baseline),
        notes_json=json.dumps(report.notes),
        spec_json=json.dumps(report.spec.to_dict()),
        classes_json=json.dumps(report.classes or []),
        n_rows=report.n_rows,
        n_features=report.n_features,
        train_rows=report.train_rows,
        test_rows=report.test_rows,
        duration_seconds=round(duration, 3),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return _to_detail(run)


def _to_detail(run: Run) -> RunDetail:
    spec = json.loads(run.spec_json or "{}")
    return RunDetail(
        id=run.id,
        dataset_id=run.dataset_id,
        target=run.target,
        task=run.task,
        status=run.status,
        winner_name=run.winner_name,
        winner_family=run.winner_family,
        primary_metric=run.primary_metric,
        primary_score=run.primary_score,
        baseline_score=run.baseline_score,
        n_rows=run.n_rows,
        n_features=run.n_features,
        train_rows=run.train_rows,
        test_rows=run.test_rows,
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
        leaderboard=[
            CandidateOut(**c) for c in json.loads(run.leaderboard_json or "[]")
        ],
        baseline=json.loads(run.baseline_json or "{}"),
        notes=json.loads(run.notes_json or "[]"),
        classes=json.loads(run.classes_json or "[]"),
        feature_names=spec.get("feature_names", []),
    )


@router.get("", response_model=list[RunOut])
def list_runs(
    db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[RunOut]:
    rows = db.scalars(
        select(Run).where(Run.owner_id == user.id).order_by(Run.created_at.desc())
    ).all()
    return [RunOut.model_validate(r) for r in rows]


@router.get("/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> RunDetail:
    return _to_detail(_owned_run(db, run_id, user))


@router.delete(
    "/{run_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response
)
def delete_run(
    run_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> Response:
    run = _owned_run(db, run_id, user)
    db.delete(run)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{run_id}/predict", response_model=PredictResponse)
def predict(
    run_id: str,
    payload: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> PredictResponse:
    """Score one row against the saved winner.

    The model is refit from the stored dataset and spec rather than
    deserialised from a pickle. That is slower, but it means a run can never
    be scored by a model that differs from the one the leaderboard describes.
    """
    run = _owned_run(db, run_id, user)
    if run.status != "completed" or not run.winner_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This run has no successful model to score with.",
        )

    ds = db.get(Dataset, run.dataset_id)
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The dataset behind this run no longer exists.",
        )

    table = parse_csv(ds.content, max_rows=settings.max_rows)
    spec = FeatureSpec.from_dict(json.loads(run.spec_json))

    # Build a one-row table from the payload, filling anything missing with
    # an empty string so the encoder treats it as a blank rather than failing.
    row = {col: "" for col in table.columns}
    for key, value in payload.row.items():
        if key in row:
            row[key] = "" if value is None else str(value)
    one = ParsedTable(columns=table.columns, rows=[row], truncated=False)

    try:
        X = transform(one, spec)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not encode the supplied row: {exc}",
        ) from exc

    # Refit the winner on the full dataset.
    report = train(
        table,
        target=run.target,
        task=run.task,
        test_fraction=0.25,
        seed=42,
    )
    model = report.fitted_model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The winning model could not be rebuilt.",
        )

    raw_pred = model.predict(X)[0]
    classes = json.loads(run.classes_json or "[]")

    if run.task == "classification":
        label = classes[int(raw_pred)] if classes else str(int(raw_pred))
        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            probs = {
                (classes[i] if i < len(classes) else str(i)): round(float(v), 4)
                for i, v in enumerate(p)
            }
        return PredictResponse(
            prediction=label,
            task=run.task,
            model=run.winner_name,
            probabilities=probs,
        )

    return PredictResponse(
        prediction=round(float(np.asarray(raw_pred).item()), 4),
        task=run.task,
        model=run.winner_name,
        probabilities=None,
    )
