"""Turn a parsed CSV into a numeric matrix, and remember how we did it.

The encoder is the part people skip and then regret. If you one-hot encode
a column at training time and forget the category list, you cannot score a
new row later. So every transform here is recorded in a `FeatureSpec` that
is stored alongside the model and replayed at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .csv_io import ParsedTable

MISSING_TOKEN = "__missing__"
MAX_CATEGORIES = 30


class DatasetError(ValueError):
    """Raised when a table cannot be turned into a trainable matrix."""


@dataclass
class ColumnSpec:
    """How one source column became one or more numeric features."""

    name: str
    kind: str  # "numeric" | "categorical"
    # For numeric columns: the fill value used for blanks.
    fill: float = 0.0
    # For categorical columns: the category list, in index order.
    categories: list[str] = field(default_factory=list)
    # For numeric columns: mean/std used to standardise. std==0 means skip.
    mean: float = 0.0
    std: float = 1.0


@dataclass
class FeatureSpec:
    """Everything needed to reproduce the transform on unseen rows."""

    columns: list[ColumnSpec]
    feature_names: list[str]

    def to_dict(self) -> dict:
        return {
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "fill": c.fill,
                    "categories": c.categories,
                    "mean": c.mean,
                    "std": c.std,
                }
                for c in self.columns
            ],
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FeatureSpec:
        return cls(
            columns=[ColumnSpec(**c) for c in raw["columns"]],
            feature_names=list(raw["feature_names"]),
        )


def _is_number(value: str) -> bool:
    if value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def infer_kind(values: list[str]) -> str:
    """A column is numeric only if every non-blank value parses as a number.

    One stray "N/A" in a column of prices makes the whole column categorical.
    That is the conservative choice: treating text as a number would mean
    inventing a value, and inventing values is how leaderboards lie.
    """
    present = [v for v in values if v != ""]
    if not present:
        return "categorical"
    numeric = sum(1 for v in present if _is_number(v))
    return "numeric" if numeric == len(present) else "categorical"


def build_spec(
    table: ParsedTable,
    target: str,
    feature_columns: list[str] | None = None,
) -> FeatureSpec:
    """Decide, from the data, how each feature column will be encoded."""
    if target not in table.columns:
        raise DatasetError(f"Target column '{target}' is not in the file.")

    candidates = feature_columns or [c for c in table.columns if c != target]
    if not candidates:
        raise DatasetError(
            "There are no feature columns left after removing the target."
        )

    specs: list[ColumnSpec] = []
    names: list[str] = []

    for col in candidates:
        if col not in table.columns:
            raise DatasetError(f"Feature column '{col}' is not in the file.")
        values = [row[col] for row in table.rows]
        kind = infer_kind(values)

        if kind == "numeric":
            present = [float(v) for v in values if v != ""]
            arr = np.asarray(present, dtype=float)
            mean = float(arr.mean()) if arr.size else 0.0
            std = float(arr.std()) if arr.size else 0.0
            specs.append(
                ColumnSpec(
                    name=col,
                    kind="numeric",
                    fill=mean,
                    mean=mean,
                    std=std if std > 1e-12 else 1.0,
                )
            )
            names.append(col)
        else:
            counts: dict[str, int] = {}
            for v in values:
                key = v if v != "" else MISSING_TOKEN
                counts[key] = counts.get(key, 0) + 1
            # Keep the most common categories; everything else collapses into
            # one bucket so a long tail cannot explode the feature count.
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            kept = [k for k, _ in ordered[:MAX_CATEGORIES]]
            specs.append(ColumnSpec(name=col, kind="categorical", categories=kept))
            names.extend(f"{col}={cat}" for cat in kept)
            names.append(f"{col}=__other__")

    return FeatureSpec(columns=specs, feature_names=names)


def transform(table: ParsedTable, spec: FeatureSpec) -> np.ndarray:
    """Apply a fitted spec to rows, producing a dense float matrix."""
    out = np.zeros((len(table.rows), len(spec.feature_names)), dtype=float)
    cursor = 0

    for col_spec in spec.columns:
        if col_spec.kind == "numeric":
            for i, row in enumerate(table.rows):
                raw = row.get(col_spec.name, "")
                value = float(raw) if _is_number(raw) else col_spec.fill
                out[i, cursor] = (value - col_spec.mean) / col_spec.std
            cursor += 1
        else:
            index = {cat: j for j, cat in enumerate(col_spec.categories)}
            width = len(col_spec.categories) + 1  # +1 for the __other__ bucket
            for i, row in enumerate(table.rows):
                raw = row.get(col_spec.name, "")
                key = raw if raw != "" else MISSING_TOKEN
                slot = index.get(key, len(col_spec.categories))
                out[i, cursor + slot] = 1.0
            cursor += width

    return out


def encode_target(
    table: ParsedTable, target: str, task: str
) -> tuple[np.ndarray, list[str] | None]:
    """Return the target vector, plus the class list for classification."""
    raw = [row[target] for row in table.rows]

    if task == "regression":
        values = []
        for v in raw:
            if not _is_number(v):
                raise DatasetError(
                    f"Target column '{target}' contains a non-numeric value "
                    f"({v!r}), so it cannot be used for regression."
                )
            values.append(float(v))
        return np.asarray(values, dtype=float), None

    classes = sorted({v for v in raw if v != ""})
    if len(classes) < 2:
        raise DatasetError(
            f"Target column '{target}' has {len(classes)} distinct value(s). "
            "Classification needs at least two."
        )
    if len(classes) > 50:
        raise DatasetError(
            f"Target column '{target}' has {len(classes)} distinct values. "
            "That looks like a continuous column — try regression instead."
        )
    index = {c: i for i, c in enumerate(classes)}
    encoded = np.asarray([index.get(v, 0) for v in raw], dtype=int)
    return encoded, classes


def detect_task(table: ParsedTable, target: str) -> str:
    """Guess the task from the target column, for the UI's default."""
    values = [row[target] for row in table.rows if row[target] != ""]
    if not values:
        return "classification"
    distinct = len(set(values))
    if distinct <= 2:
        return "classification"
    if distinct <= 20 and distinct / max(len(values), 1) < 0.2:
        return "classification"
    return "regression"


def stratified_split(
    y: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so each class keeps its proportion in both halves.

    A plain random split can put every example of a rare class in train,
    which makes the test score meaningless. Stratifying per class avoids it.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 10:
        raise DatasetError("At least 10 rows are needed to split train and test.")

    test_idx: list[int] = []
    for cls in np.unique(y):
        members = np.flatnonzero(y == cls)
        rng.shuffle(members)
        take = max(1, int(round(len(members) * test_fraction)))
        take = min(take, len(members) - 1) if len(members) > 1 else 0
        test_idx.extend(members[:take].tolist())

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    train_idx = np.flatnonzero(~test_mask)
    test_idx_arr = np.flatnonzero(test_mask)

    if len(train_idx) == 0 or len(test_idx_arr) == 0:
        raise DatasetError("The split produced an empty train or test set.")

    return train_idx, test_idx_arr
