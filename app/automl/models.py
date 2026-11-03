"""Four model families, written in NumPy.

No scikit-learn, no XGBoost, no autograd. Each class below implements
`fit` and `predict` against plain arrays, which means the leaderboard is
comparing implementations you can read rather than library calls you can't.

The four families are chosen to fail in different ways, which is the point
of an AutoML leaderboard: a linear model that wins means the signal is
linear; a forest that wins means it isn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Linear models
# --------------------------------------------------------------------------


@dataclass
class LogisticRegression:
    """Multinomial logistic regression trained with full-batch gradient descent.

    L2 regularisation is applied to the weights but never to the bias, which
    is the standard convention: shrinking the bias would bias every
    prediction toward the majority class.
    """

    learning_rate: float = 0.5
    epochs: int = 300
    l2: float = 0.01
    n_classes: int = 2
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    loss_history: list[float] = field(default_factory=list)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        n, d = X.shape
        k = self.n_classes
        self.weights = np.zeros((d, k), dtype=float)
        self.bias = np.zeros(k, dtype=float)
        self.loss_history = []

        Y = np.zeros((n, k), dtype=float)
        Y[np.arange(n), y.astype(int)] = 1.0

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            P = self._softmax(z)

            eps = 1e-12
            loss = float(-np.sum(Y * np.log(P + eps)) / n)
            loss += float(self.l2 * np.sum(self.weights**2) / 2)
            self.loss_history.append(loss)

            grad = (P - Y) / n
            self.weights -= self.learning_rate * (X.T @ grad + self.l2 * self.weights)
            self.bias -= self.learning_rate * grad.sum(axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted.")
        return self._softmax(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class RidgeRegression:
    """Closed-form ridge regression.

    Solved as (X'X + aI)^-1 X'y with the intercept handled by centring, so
    the penalty never touches the bias term.
    """

    alpha: float = 1.0
    coef: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        n, d = X.shape
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        Xc = X - x_mean
        yc = y - y_mean

        A = Xc.T @ Xc + self.alpha * np.eye(d)
        b = Xc.T @ yc
        try:
            self.coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:  # pragma: no cover - alpha>0 keeps A invertible
            self.coef = np.linalg.lstsq(A, b, rcond=None)[0]
        self.intercept = y_mean - float(x_mean @ self.coef)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Model has not been fitted.")
        return X @ self.coef + self.intercept


# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------


@dataclass
class _Node:
    """One split, or a leaf when `feature` is -1."""

    feature: int = -1
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None
    value: float = 0.0  # leaf prediction (class index or regression value)


@dataclass
class DecisionTree:
    """CART with gini impurity for classification and variance for regression.

    Splits are found by sorting each candidate feature once and scanning
    prefix sums, which is O(d * n log n) per node instead of the naive
    O(d * n^2) threshold sweep.
    """

    max_depth: int = 6
    min_samples_leaf: int = 3
    task: str = "classification"
    n_classes: int = 2
    max_features: int | None = None
    rng: np.random.Generator | None = None
    root: _Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        self.root = self._build(X, y, depth=0)
        return self

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.task == "regression":
            return float(y.mean())
        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        return float(np.argmax(counts))

    def _impurity(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        if self.task == "regression":
            return float(y.var())
        counts = np.bincount(y.astype(int), minlength=self.n_classes).astype(float)
        p = counts / counts.sum()
        return float(1.0 - np.sum(p**2))

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=self._leaf_value(y))

        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or self._impurity(y) <= 1e-12
        ):
            return node

        n, d = X.shape
        if self.max_features is not None and self.max_features < d:
            rng = self.rng or np.random.default_rng(0)
            features = rng.choice(d, size=self.max_features, replace=False)
        else:
            features = np.arange(d)

        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        parent_impurity = self._impurity(y)
        total = len(y)

        for f in features:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            ys = y[order]

            # Only positions where the value actually changes are candidates.
            distinct = np.flatnonzero(np.diff(xs) > 1e-12)
            if distinct.size == 0:
                continue

            for i in distinct:
                left_n = i + 1
                right_n = total - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                left_y = ys[:left_n]
                right_y = ys[left_n:]
                weighted = (
                    left_n * self._impurity(left_y) + right_n * self._impurity(right_y)
                ) / total
                gain = parent_impurity - weighted

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(f)
                    best_threshold = float((xs[i] + xs[i + 1]) / 2.0)

        if best_feature < 0:
            return node

        mask = X[:, best_feature] <= best_threshold
        if mask.all() or not mask.any():
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, row: np.ndarray, node: _Node) -> float:
        while node.feature >= 0:
            node = node.left if row[node.feature] <= node.threshold else node.right
            if node is None:  # pragma: no cover - defensive
                return 0.0
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model has not been fitted.")
        return np.asarray([self._predict_one(row, self.root) for row in X])


# --------------------------------------------------------------------------
# Random forest
# --------------------------------------------------------------------------


@dataclass
class RandomForest:
    """Bagged trees with per-split feature subsampling.

    Two sources of randomness (row bootstrap, feature subset) is what makes
    the ensemble decorrelate. Without the feature subset the trees would be
    near-identical and averaging them would buy almost nothing.
    """

    n_estimators: int = 12
    max_depth: int = 6
    min_samples_leaf: int = 3
    task: str = "classification"
    n_classes: int = 2
    max_features: int | None = None
    seed: int = 0
    trees: list[DecisionTree] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        mf = self.max_features or max(1, int(np.sqrt(d)))
        self.trees = []

        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                task=self.task,
                n_classes=self.n_classes,
                max_features=mf,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Model has not been fitted.")
        preds = np.vstack([t.predict(X) for t in self.trees])

        if self.task == "regression":
            return preds.mean(axis=0)

        # Majority vote per row.
        out = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            votes = np.bincount(preds[:, i].astype(int), minlength=self.n_classes)
            out[i] = int(np.argmax(votes))
        return out


# --------------------------------------------------------------------------
# Candidate definitions
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """A named model plus the hyperparameters that produced it."""

    name: str
    family: str
    params: dict
    model: object


def build_candidates(
    task: str, n_classes: int, n_features: int, seed: int
) -> list[Candidate]:
    """The search space. Small on purpose — every entry is a defensible choice."""
    out: list[Candidate] = []

    if task == "classification":
        for lr, l2 in ((0.5, 0.01), (0.3, 0.1)):
            out.append(
                Candidate(
                    name=f"logistic_regression(lr={lr}, l2={l2})",
                    family="logistic_regression",
                    params={"learning_rate": lr, "l2": l2, "epochs": 300},
                    model=LogisticRegression(
                        learning_rate=lr, l2=l2, epochs=300, n_classes=n_classes
                    ),
                )
            )
        for depth in (4, 8):
            out.append(
                Candidate(
                    name=f"decision_tree(depth={depth})",
                    family="decision_tree",
                    params={"max_depth": depth, "min_samples_leaf": 3},
                    model=DecisionTree(
                        max_depth=depth,
                        min_samples_leaf=3,
                        task="classification",
                        n_classes=n_classes,
                    ),
                )
            )
        for n_est, depth in ((12, 6), (24, 10)):
            out.append(
                Candidate(
                    name=f"random_forest(trees={n_est}, depth={depth})",
                    family="random_forest",
                    params={"n_estimators": n_est, "max_depth": depth},
                    model=RandomForest(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_leaf=2,
                        task="classification",
                        n_classes=n_classes,
                        seed=seed,
                    ),
                )
            )
    else:
        for alpha in (0.1, 1.0, 10.0):
            out.append(
                Candidate(
                    name=f"ridge(alpha={alpha})",
                    family="ridge",
                    params={"alpha": alpha},
                    model=RidgeRegression(alpha=alpha),
                )
            )
        for depth in (4, 8):
            out.append(
                Candidate(
                    name=f"decision_tree(depth={depth})",
                    family="decision_tree",
                    params={"max_depth": depth, "min_samples_leaf": 3},
                    model=DecisionTree(
                        max_depth=depth, min_samples_leaf=3, task="regression"
                    ),
                )
            )
        for n_est, depth in ((12, 6), (24, 10)):
            out.append(
                Candidate(
                    name=f"random_forest(trees={n_est}, depth={depth})",
                    family="random_forest",
                    params={"n_estimators": n_est, "max_depth": depth},
                    model=RandomForest(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_leaf=2,
                        task="regression",
                        seed=seed,
                    ),
                )
            )

    return out
