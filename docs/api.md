# API reference

Base path: `/api`. All responses are JSON. Interactive docs at `/api/docs`.

## Authentication

Every endpoint except `/api/health`, `/api/auth/register`, `/api/auth/login`
and `/api/samples*` requires:

```
Authorization: Bearer <access_token>
```

Tokens are HS256, expire after 12 hours, and carry only the user id in
`sub`.

---

## Health

### `GET /api/health`

No auth.

```json
{ "status": "ok", "version": "1.0.0", "environment": "development" }
```

---

## Auth

### `POST /api/auth/register`

Rate limited. Returns **201**.

```json
{ "email": "you@example.com", "name": "Your Name", "password": "at-least-8" }
```

Password rules: 8–128 chars, not all digits, not in a small common-password
list. Violations return **422** with the reason.

**201**

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "user": {
    "id": "u_8f3a...",
    "email": "you@example.com",
    "name": "Your Name",
    "role": "analyst",
    "created_at": "2026-11-04T09:12:33Z"
  }
}
```

| Status | When |
|---|---|
| 201 | Created |
| 409 | Email already registered |
| 422 | Validation failed |
| 429 | Rate limit exceeded |

### `POST /api/auth/login`

Rate limited. Returns **200** with the same body shape as register.

| Status | When |
|---|---|
| 200 | OK |
| 401 | Wrong email or password (same message for both — no enumeration) |
| 429 | Rate limit exceeded |

### `GET /api/auth/me`

Returns the `UserOut` object.

---

## Datasets

### `POST /api/datasets`

`multipart/form-data`, field name `file`. Returns **201**.

| Status | When |
|---|---|
| 201 | Parsed and stored |
| 400 | Not a `.csv`, empty, or unparseable |
| 413 | Larger than `MAX_UPLOAD_BYTES` (5 MB) |
| 422 | Fewer than 20 data rows, or more than 200 columns |

**201**

```json
{
  "id": "ds_1a2b...",
  "name": "iris",
  "filename": "iris.csv",
  "n_rows": 150,
  "n_columns": 5,
  "created_at": "2026-11-04T09:14:02Z"
}
```

### `GET /api/datasets`

Returns `DatasetOut[]`, newest first.

### `GET /api/datasets/{id}`

Returns `DatasetDetail` — the list above plus:

```json
{
  "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
  "preview": [
    { "sepal_length": "5.1", "sepal_width": "3.5", "petal_length": "1.4",
      "petal_width": "0.2", "species": "setosa" }
  ],
  "suggested_target": "species",
  "suggested_task": "classification"
}
```

`preview` is the first 8 rows, values as strings exactly as they appeared in
the file.

| Status | When |
|---|---|
| 200 | OK |
| 404 | Not found, or owned by another user |

### `DELETE /api/datasets/{id}`

Returns **204** with an empty body. Cascades to that dataset's runs.

---

## Runs

### `POST /api/runs`

Trains every candidate and returns the full report. **Synchronous** — the
request blocks until the search finishes. Returns **201**.

```json
{
  "dataset_id": "ds_1a2b...",
  "target": "species",
  "task": "classification",
  "test_fraction": 0.25,
  "seed": 42,
  "feature_columns": null
}
```

| Field | Default | Notes |
|---|---|---|
| `target` | required | Must be a column in the dataset |
| `task` | required | `classification` or `regression` |
| `test_fraction` | `0.25` | Must be `> 0.05` and `< 0.6` |
| `seed` | `42` | Split and forest randomness |
| `feature_columns` | `null` | `null` = every column except the target |

| Status | When |
|---|---|
| 201 | Trained |
| 404 | Dataset not found or not yours |
| 422 | Unknown target, target not numeric for regression, or more than 50 classes |

**201** — `RunDetail`:

```json
{
  "id": "run_9c8d...",
  "dataset_id": "ds_1a2b...",
  "target": "species",
  "task": "classification",
  "status": "completed",
  "winner_name": "decision_tree(depth=4)",
  "winner_family": "tree",
  "primary_metric": "accuracy",
  "primary_score": 1.0,
  "baseline_score": 0.3333,
  "n_rows": 150,
  "n_features": 4,
  "train_rows": 112,
  "test_rows": 38,
  "duration_seconds": 0.42,
  "created_at": "2026-11-04T09:15:10Z",
  "leaderboard": [
    {
      "name": "decision_tree(depth=4)",
      "family": "tree",
      "params": { "max_depth": 4, "min_samples_split": 2 },
      "scores": { "accuracy": 1.0, "macro_f1": 1.0 },
      "train_seconds": 0.011,
      "error": null
    }
  ],
  "baseline": { "accuracy": 0.3333, "macro_f1": 0.1667 },
  "notes": [
    "The top 4 models are within 0.01 of each other — prefer the simplest."
  ],
  "classes": ["setosa", "versicolor", "virginica"],
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
}
```

`leaderboard` is sorted best-first. A candidate that raised during fit
appears with `error` set and empty `scores` — one bad candidate never fails
the whole run.

`primary_metric` is `accuracy` for classification and `r2` for regression.

### `GET /api/runs`

Returns `RunOut[]` (the summary shape, without `leaderboard`), newest first.

### `GET /api/runs/{id}`

Returns the full `RunDetail`.

### `DELETE /api/runs/{id}`

Returns **204**.

### `POST /api/runs/{id}/predict`

Scores one row against the run's winner. Returns **200**.

```json
{
  "row": {
    "sepal_length": "5.1",
    "sepal_width": "3.5",
    "petal_length": "1.4",
    "petal_width": "0.2"
  }
}
```

Values may be strings or numbers; they go through the same encoder the
model was trained with. Missing keys are filled with the column mean (or the
`__missing__` category). Extra keys are ignored.

**200** — classification:

```json
{
  "prediction": "setosa",
  "task": "classification",
  "model": "decision_tree(depth=4)",
  "probabilities": null
}
```

`probabilities` is `null` for trees and forests, which do not expose
calibrated class probabilities. Logistic regression returns a map:

```json
{ "setosa": 0.98, "versicolor": 0.02, "virginica": 0.0 }
```

**200** — regression:

```json
{ "prediction": 72.4418, "task": "regression", "model": "ridge(alpha=10.0)", "probabilities": null }
```

| Status | When |
|---|---|
| 200 | Scored |
| 404 | Run not found or not yours |
| 409 | Run did not complete successfully |

---

## Samples

### `GET /api/samples`

No auth. Returns the three bundled datasets:

```json
[
  {
    "key": "iris",
    "title": "Iris species",
    "task": "classification",
    "target": "species",
    "blurb": "150 flowers, 3 species, 4 measurements. The hello-world of classification.",
    "available": true
  }
]
```

`available` is `false` when the CSV has not been generated yet — run
`python scripts/make_samples.py`.

### `GET /api/samples/{key}`

No auth. Returns the raw CSV as `text/csv` with a
`Content-Disposition: attachment` header. Keys: `iris`, `churn`, `exam`.

| Status | When |
|---|---|
| 200 | OK |
| 404 | Unknown key |

---

## Error shape

Every error uses the same envelope:

```json
{ "detail": "Dataset not found." }
```

Validation errors (422) use FastAPI's standard list form:

```json
{
  "detail": [
    { "loc": ["body", "task"], "msg": "Input should be 'classification' or 'regression'", "type": "literal_error" }
  ]
}
```

Unhandled exceptions return **500** with `{"detail": "Internal server error."}`
and log the traceback server-side. No stack trace ever reaches the client.

## Rate limits

| Endpoint | Limit |
|---|---|
| `POST /api/auth/register` | 5 / minute per IP |
| `POST /api/auth/login` | 10 / minute per IP |

Counters are in-process. Behind multiple workers the effective limit is
`limit × workers` — see `ARCHITECTURE.md` for the Redis fix.
