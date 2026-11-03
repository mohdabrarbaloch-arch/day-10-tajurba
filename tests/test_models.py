"""Model correctness.

These tests check the properties that matter rather than exact numbers:
a linear model must recover a linear relationship, a tree must split a
step function, and a forest must not be worse than a single tree on
average. Exact-value assertions on trained models are brittle and would
break on any legitimate refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.automl.models import (
    DecisionTree,
    LogisticRegression,
    RandomForest,
    RidgeRegression,
    build_candidates,
)


def _linear_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 4.0
    return X, y


def _separable_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.6, (n // 2, 2)), rng.normal(3, 0.6, (n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_ridge_recovers_linear_coefficients():
    """With no noise, ridge should recover the true coefficients closely."""
    X, y = _linear_data()
    model = RidgeRegression(alpha=0.01).fit(X, y)
    assert model.coef[0] == pytest.approx(2.0, abs=0.05)
    assert model.coef[1] == pytest.approx(-1.5, abs=0.05)
    assert model.intercept == pytest.approx(4.0, abs=0.05)


def test_ridge_predictions_are_accurate():
    X, y = _linear_data()
    model = RidgeRegression(alpha=0.1).fit(X, y)
    pred = model.predict(X)
    assert np.sqrt(np.mean((y - pred) ** 2)) < 0.1


def test_ridge_alpha_shrinks_coefficients():
    """A larger penalty must produce smaller weights — that is the whole point."""
    X, y = _linear_data()
    small = RidgeRegression(alpha=0.01).fit(X, y)
    large = RidgeRegression(alpha=1000.0).fit(X, y)
    assert np.linalg.norm(large.coef) < np.linalg.norm(small.coef)


def test_logistic_separates_two_clusters():
    X, y = _separable_data()
    model = LogisticRegression(learning_rate=0.5, epochs=400, n_classes=2).fit(X, y)
    pred = model.predict(X)
    assert np.mean(pred == y) > 0.95


def test_logistic_probabilities_sum_to_one():
    X, y = _separable_data()
    model = LogisticRegression(epochs=100, n_classes=2).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_logistic_loss_decreases():
    X, y = _separable_data()
    model = LogisticRegression(epochs=200, n_classes=2).fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1]


def test_logistic_handles_three_classes():
    rng = np.random.default_rng(3)
    X = np.vstack(
        [
            rng.normal(0, 0.5, (60, 2)),
            rng.normal(4, 0.5, (60, 2)),
            rng.normal(8, 0.5, (60, 2)),
        ]
    )
    y = np.array([0] * 60 + [1] * 60 + [2] * 60)
    model = LogisticRegression(epochs=400, n_classes=3).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.9


def test_tree_learns_a_step_function():
    """A tree should nail a threshold rule that a linear model cannot."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = (X[:, 0] > 50).astype(int)
    tree = DecisionTree(max_depth=3, min_samples_leaf=2, n_classes=2).fit(X, y)
    assert np.mean(tree.predict(X) == y) > 0.95


def test_tree_regression_fits_nonlinear_target():
    X = np.linspace(-3, 3, 200).reshape(-1, 1)
    y = X[:, 0] ** 2
    tree = DecisionTree(max_depth=6, min_samples_leaf=2, task="regression").fit(X, y)
    pred = tree.predict(X)
    assert np.sqrt(np.mean((y - pred) ** 2)) < 1.0


def test_tree_depth_limit_is_respected():
    """A depth-1 tree can only make one split, so it cannot fit a 4-step
    function exactly — this proves max_depth is actually enforced."""
    X = np.arange(80, dtype=float).reshape(-1, 1)
    y = (X[:, 0] // 20).astype(int)
    shallow = DecisionTree(max_depth=1, min_samples_leaf=2, n_classes=4).fit(X, y)
    deep = DecisionTree(max_depth=6, min_samples_leaf=2, n_classes=4).fit(X, y)
    assert np.mean(shallow.predict(X) == y) < np.mean(deep.predict(X) == y)


def test_forest_beats_single_tree_on_noisy_data():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(0, 0.8, 300) > 0).astype(int)

    tree = DecisionTree(max_depth=4, min_samples_leaf=3, n_classes=2).fit(X, y)
    forest = RandomForest(
        n_estimators=15, max_depth=4, min_samples_leaf=3, n_classes=2, seed=1
    ).fit(X, y)

    tree_acc = np.mean(tree.predict(X) == y)
    forest_acc = np.mean(forest.predict(X) == y)
    assert forest_acc >= tree_acc - 0.02


def test_forest_regression_averages_trees():
    X, y = _linear_data()
    forest = RandomForest(n_estimators=10, max_depth=5, task="regression", seed=2).fit(
        X, y
    )
    pred = forest.predict(X)
    assert pred.shape == y.shape
    assert np.sqrt(np.mean((y - pred) ** 2)) < 1.5


def test_unfitted_models_raise():
    with pytest.raises(RuntimeError):
        RidgeRegression().predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        DecisionTree().predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        RandomForest().predict(np.zeros((2, 2)))


def test_candidate_space_matches_task():
    cls = build_candidates("classification", n_classes=3, n_features=4, seed=0)
    reg = build_candidates("regression", n_classes=0, n_features=4, seed=0)
    assert len(cls) >= 6
    assert len(reg) >= 6
    assert all("ridge" in c.name for c in reg if c.family == "ridge")
    assert not any(c.family == "ridge" for c in cls)
