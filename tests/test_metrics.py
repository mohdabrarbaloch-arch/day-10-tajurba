"""Metric correctness. These are the numbers the whole product reports."""

from __future__ import annotations

import numpy as np
import pytest

from app.automl import metrics


def test_accuracy_perfect_and_zero():
    y = np.array([0, 1, 2, 1])
    assert metrics.accuracy(y, y) == 1.0
    assert metrics.accuracy(y, np.array([1, 0, 0, 0])) == 0.0


def test_accuracy_known_fraction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert metrics.accuracy(y_true, y_pred) == 0.75


def test_confusion_matrix_orientation():
    """Rows must be truth, columns prediction — swapping them silently
    inverts precision and recall."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = metrics.confusion_matrix(y_true, y_pred, 2)
    assert cm[0, 0] == 1  # true 0, predicted 0
    assert cm[0, 1] == 1  # true 0, predicted 1
    assert cm[1, 0] == 1  # true 1, predicted 0
    assert cm[1, 1] == 1


def test_per_class_f1_hand_computed():
    """Class 0: tp=1, fp=0, fn=1 -> p=1.0, r=0.5, f1=0.6667."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    scores = metrics.per_class_f1(y_true, y_pred, 2)
    assert scores[0]["precision"] == 1.0
    assert scores[0]["recall"] == 0.5
    assert scores[0]["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert scores[0]["support"] == 2


def test_macro_f1_punishes_ignoring_a_class():
    """A model that always predicts the majority class should score badly
    on macro F1 even though its accuracy looks respectable."""
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.zeros(10, dtype=int)
    assert metrics.accuracy(y_true, y_pred) == 0.8
    assert metrics.macro_f1(y_true, y_pred, 2) < 0.5


def test_rmse_and_mae_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert metrics.rmse(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))
    assert metrics.mae(y_true, y_pred) == pytest.approx(2 / 3)


def test_r2_perfect_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.r2(y, y) == pytest.approx(1.0)


def test_r2_can_be_negative():
    """A model worse than the mean must show a negative R^2, not be clipped."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 10.0, 10.0])
    assert metrics.r2(y_true, y_pred) < 0


def test_r2_constant_target_returns_zero():
    y = np.array([5.0, 5.0, 5.0])
    assert metrics.r2(y, np.array([5.0, 5.0, 5.0])) == 0.0


def test_empty_arrays_do_not_crash():
    empty = np.array([])
    assert metrics.accuracy(empty, empty) == 0.0
    assert metrics.rmse(empty, empty) == 0.0
    assert metrics.r2(empty, empty) == 0.0


def test_primary_metric_names():
    assert metrics.primary_metric_name("classification") == "accuracy"
    assert metrics.primary_metric_name("regression") == "r2"
