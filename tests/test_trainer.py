"""The training pipeline: encoding, splitting, baselines, leaderboard."""

from __future__ import annotations

import numpy as np
import pytest

from app.automl.csv_io import CsvError, parse_csv
from app.automl.dataset import (
    DatasetError,
    build_spec,
    detect_task,
    encode_target,
    infer_kind,
    stratified_split,
    transform,
)
from app.automl.trainer import train


# ── CSV parsing ──────────────────────────────────────────────────────


def test_parse_csv_basic(iris_csv):
    table = parse_csv(iris_csv)
    assert len(table.columns) == 5
    assert len(table.rows) == 150
    assert table.columns[0] == "species"


def test_parse_csv_rejects_empty():
    with pytest.raises(CsvError, match="empty"):
        parse_csv("")


def test_parse_csv_rejects_too_few_rows():
    with pytest.raises(CsvError, match="At least 20"):
        parse_csv("a,b\n1,2\n3,4\n")


def test_parse_csv_rejects_duplicate_headers():
    rows = "\n".join(f"{i},{i}" for i in range(25))
    with pytest.raises(CsvError, match="duplicate"):
        parse_csv(f"x,x\n{rows}\n")


def test_parse_csv_skips_ragged_rows():
    """A row with the wrong column count is dropped, not padded."""
    body = "\n".join(f"{i},{i * 2}" for i in range(25))
    text = f"a,b\n{body}\n99,99,99\n"
    table = parse_csv(text)
    assert len(table.rows) == 25


def test_parse_csv_handles_semicolons():
    body = "\n".join(f"{i};{i * 2}" for i in range(25))
    table = parse_csv(f"a;b\n{body}\n")
    assert table.columns == ["a", "b"]
    assert len(table.rows) == 25


# ── Type inference and encoding ──────────────────────────────────────


def test_infer_kind_numeric_and_categorical():
    assert infer_kind(["1", "2", "3.5"]) == "numeric"
    assert infer_kind(["a", "b", "c"]) == "categorical"
    # One non-numeric value makes the whole column categorical.
    assert infer_kind(["1", "2", "N/A"]) == "categorical"


def test_infer_kind_ignores_blanks():
    assert infer_kind(["1", "", "3"]) == "numeric"


def test_transform_produces_expected_width(iris_csv):
    table = parse_csv(iris_csv)
    spec = build_spec(table, "species")
    X = transform(table, spec)
    assert X.shape == (150, len(spec.feature_names))
    assert X.shape[1] == 4  # four numeric features, no one-hot needed


def test_transform_one_hot_encodes_categoricals(churn_csv):
    table = parse_csv(churn_csv)
    spec = build_spec(table, "churned")
    X = transform(table, spec)
    # contract (3) + city (5) each get their categories plus an __other__ bucket
    assert X.shape[1] == 3 + 3 + 1 + 5 + 1


def test_transform_is_replayable(churn_csv):
    """The same spec applied twice must give identical output — this is what
    makes prediction reproducible."""
    table = parse_csv(churn_csv)
    spec = build_spec(table, "churned")
    assert np.allclose(transform(table, spec), transform(table, spec))


def test_spec_round_trips_through_dict(iris_csv):
    from app.automl.dataset import FeatureSpec

    table = parse_csv(iris_csv)
    spec = build_spec(table, "species")
    restored = FeatureSpec.from_dict(spec.to_dict())
    assert restored.feature_names == spec.feature_names
    assert np.allclose(transform(table, spec), transform(table, restored))


def test_encode_target_classification(iris_csv):
    table = parse_csv(iris_csv)
    y, classes = encode_target(table, "species", "classification")
    assert classes == ["setosa", "versicolor", "virginica"]
    assert set(np.unique(y)) == {0, 1, 2}


def test_encode_target_regression_rejects_text():
    text = "y,x\n" + "\n".join(f"abc,{i}" for i in range(25))
    table = parse_csv(text)
    with pytest.raises(DatasetError, match="non-numeric"):
        encode_target(table, "y", "regression")


def test_detect_task_guesses_correctly(iris_csv, exam_csv):
    assert detect_task(parse_csv(iris_csv), "species") == "classification"
    assert detect_task(parse_csv(exam_csv), "final_score") == "regression"


# ── Splitting ────────────────────────────────────────────────────────


def test_stratified_split_preserves_class_ratio():
    y = np.array([0] * 80 + [1] * 20)
    train_idx, test_idx = stratified_split(y, 0.25, seed=1)
    assert len(train_idx) + len(test_idx) == 100
    # The rare class must appear in both halves.
    assert 1 in y[test_idx]
    assert 1 in y[train_idx]


def test_stratified_split_is_deterministic():
    y = np.array([0, 1] * 50)
    a = stratified_split(y, 0.25, seed=7)
    b = stratified_split(y, 0.25, seed=7)
    assert np.array_equal(a[0], b[0])


# ── Full training ────────────────────────────────────────────────────


def test_train_classification_beats_baseline(iris_csv):
    report = train(parse_csv(iris_csv), "species", "classification")
    assert report.winner is not None
    assert report.winner.scores["accuracy"] > report.baseline["accuracy"]
    assert report.winner.scores["accuracy"] > 0.85


def test_train_reports_baseline_for_every_run(iris_csv):
    report = train(parse_csv(iris_csv), "species", "classification")
    assert "accuracy" in report.baseline
    assert report.baseline_label


def test_train_regression_produces_positive_r2(exam_csv):
    report = train(parse_csv(exam_csv), "final_score", "regression")
    assert report.winner is not None
    assert report.winner.scores["r2"] > 0.4
    assert report.winner.scores["r2"] > report.baseline["r2"]


def test_train_leaderboard_is_sorted(iris_csv):
    report = train(parse_csv(iris_csv), "species", "classification")
    scored = [r for r in report.leaderboard if r.error is None]
    accs = [r.scores["accuracy"] for r in scored]
    assert accs == sorted(accs, reverse=True)


def test_train_returns_fitted_model(iris_csv):
    report = train(parse_csv(iris_csv), "species", "classification")
    assert report.fitted_model is not None
    assert hasattr(report.fitted_model, "predict")


def test_train_rejects_missing_target(iris_csv):
    with pytest.raises(DatasetError, match="not in the file"):
        train(parse_csv(iris_csv), "nope", "classification")


def test_train_rejects_bad_task(iris_csv):
    with pytest.raises(DatasetError, match="task must be"):
        train(parse_csv(iris_csv), "species", "clustering")


def test_train_notes_flag_a_tie(iris_csv):
    """Iris is easy enough that several models hit 1.0 — the report should
    say so rather than pretending there is a clear winner."""
    report = train(parse_csv(iris_csv), "species", "classification")
    assert isinstance(report.notes, list)
