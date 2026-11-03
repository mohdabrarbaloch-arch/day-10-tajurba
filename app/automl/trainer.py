"""Run the search, score every candidate, and pick a winner.

The important design decision here is that the leaderboard is scored on a
held-out split the models never saw, and the baseline is reported next to
every score. A 0.69 accuracy means nothing until you know the majority
class gets 0.63.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .csv_io import ParsedTable
from .dataset import (
    DatasetError,
    FeatureSpec,
    build_spec,
    encode_target,
    stratified_split,
    transform,
)
from .models import build_candidates


@dataclass
class CandidateResult:
    name: str
    family: str
    params: dict
    scores: dict[str, float]
    train_seconds: float
    error: str | None = None


@dataclass
class TrainingReport:
    task: str
    target: str
    n_rows: int
    n_features: int
    n_classes: int
    classes: list[str] | None
    train_rows: int
    test_rows: int
    baseline: dict[str, float]
    baseline_label: str
    leaderboard: list[CandidateResult]
    winner: CandidateResult | None
    spec: FeatureSpec
    fitted_model: object | None = None
    notes: list[str] = field(default_factory=list)


def _baseline(
    task: str, y_train: np.ndarray, y_test: np.ndarray, n_classes: int
) -> tuple[dict[str, float], str]:
    """The dumbest defensible predictor, scored the same way as the models.

    Classification: always predict the most common training class.
    Regression: always predict the training mean.
    """
    if task == "classification":
        counts = np.bincount(y_train.astype(int), minlength=n_classes)
        majority = int(np.argmax(counts))
        pred = np.full_like(y_test, majority)
        label = f"always predict class {majority}"
        return metrics.score_predictions(task, y_test, pred, n_classes), label

    mean = float(y_train.mean())
    pred = np.full_like(y_test, mean, dtype=float)
    label = f"always predict the mean ({mean:.3f})"
    return metrics.score_predictions(task, y_test, pred, n_classes), label


def train(
    table: ParsedTable,
    target: str,
    task: str,
    test_fraction: float = 0.25,
    seed: int = 42,
    feature_columns: list[str] | None = None,
) -> TrainingReport:
    """Fit every candidate, score on the holdout, return the full report."""
    if task not in ("classification", "regression"):
        raise DatasetError("task must be 'classification' or 'regression'.")

    spec = build_spec(table, target, feature_columns)
    X = transform(table, spec)
    y, classes = encode_target(table, target, task)
    n_classes = len(classes) if classes else 0

    if X.shape[1] == 0:
        raise DatasetError("The feature matrix is empty after encoding.")

    train_idx, test_idx = stratified_split(y, test_fraction, seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    baseline_scores, baseline_label = _baseline(task, y_train, y_test, n_classes)

    results: list[CandidateResult] = []
    for cand in build_candidates(task, n_classes, X.shape[1], seed):
        started = time.perf_counter()
        try:
            cand.model.fit(X_train, y_train)
            pred = cand.model.predict(X_test)
            scores = metrics.score_predictions(task, y_test, pred, n_classes)
            elapsed = time.perf_counter() - started
            results.append(
                CandidateResult(
                    name=cand.name,
                    family=cand.family,
                    params=cand.params,
                    scores=scores,
                    train_seconds=round(elapsed, 3),
                )
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one bad candidate must not kill the run
            elapsed = time.perf_counter() - started
            results.append(
                CandidateResult(
                    name=cand.name,
                    family=cand.family,
                    params=cand.params,
                    scores={},
                    train_seconds=round(elapsed, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    primary = metrics.primary_metric_name(task)
    scored = [r for r in results if r.error is None and primary in r.scores]
    scored.sort(key=lambda r: r.scores[primary], reverse=True)

    # Keep the returned leaderboard in rank order too, so callers do not have
    # to re-sort. Failed candidates sink to the bottom.
    results.sort(
        key=lambda r: (
            r.error is not None,
            -r.scores.get(primary, float("-inf")) if r.error is None else 0.0,
        )
    )

    notes: list[str] = []
    if not scored:
        notes.append("Every candidate failed to train. See the error column.")
    else:
        best = scored[0]
        if best.scores[primary] <= baseline_scores[primary]:
            notes.append(
                f"The best model ({best.name}) did not beat the baseline "
                f"({best.scores[primary]:.4f} vs {baseline_scores[primary]:.4f}). "
                "That usually means the features carry no signal for this target."
            )
        if len(scored) > 1:
            gap = scored[0].scores[primary] - scored[1].scores[primary]
            if gap < 0.01:
                notes.append(
                    f"The top two models are within {gap:.4f} of each other. "
                    "Treat the winner as a tie and prefer the simpler one."
                )

    winner = scored[0] if scored else None
    fitted = None
    if winner is not None:
        # Refit the winner on everything, so the saved model has seen all rows.
        for cand in build_candidates(task, n_classes, X.shape[1], seed):
            if cand.name == winner.name:
                cand.model.fit(X, y)
                fitted = cand.model
                break

    return TrainingReport(
        task=task,
        target=target,
        n_rows=len(table.rows),
        n_features=X.shape[1],
        n_classes=n_classes,
        classes=classes,
        train_rows=len(train_idx),
        test_rows=len(test_idx),
        baseline=baseline_scores,
        baseline_label=baseline_label,
        leaderboard=results,
        winner=winner,
        spec=spec,
        fitted_model=fitted,
        notes=notes,
    )
