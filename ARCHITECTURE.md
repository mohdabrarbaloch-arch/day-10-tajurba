# Architecture

## The one-sentence version

A FastAPI service that turns an uploaded CSV into a numeric matrix, trains
four model families against it, and returns a leaderboard where every score
is paired with a baseline — with the encoder saved so predictions replay the
exact same transform.

## System diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (vanilla JS SPA, no build step)                         │
│  index.html · styles.css · app.js                                │
│  localStorage: JWT + user                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ fetch, Bearer token
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI (app/main.py)                                           │
│                                                                  │
│  middleware:  security headers → CORS → global error handler     │
│                                                                  │
│  routers/                                                        │
│    auth.py      register · login · me        [rate limited]      │
│    datasets.py  upload · list · detail · delete                  │
│    runs.py      train · list · detail · predict · delete         │
│    samples.py   the three bundled CSVs                           │
│                                                                  │
│  security.py    bcrypt hash · JWT issue/verify · current_user    │
│  ratelimit.py   in-process sliding window, as a dependency       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  automl/  — pure functions over NumPy, no framework imports      │
│                                                                  │
│  csv_io.py    parse_csv()      → ParsedTable                     │
│  dataset.py   build_spec()     → FeatureSpec  (the saved encoder)│
│               transform()      → X : (n, d) float                │
│               encode_target()  → y, classes                      │
│               stratified_split() → train_idx, test_idx           │
│  models.py    LogisticRegression · RidgeRegression               │
│               DecisionTree · RandomForest                        │
│  metrics.py   accuracy · macro_f1 · r2 · rmse · mae              │
│  trainer.py   train() → TrainingReport (leaderboard + winner)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  SQLAlchemy 2.0  →  SQLite (dev)  |  PostgreSQL 16 (compose)     │
│                                                                  │
│  users      id · email · name · password_hash · role             │
│  datasets   id · owner_id · content (raw CSV) · n_rows · columns │
│  runs       id · owner_id · dataset_id · target · task           │
│             leaderboard_json · spec_json · classes_json          │
│             winner_name · primary_score · baseline_score         │
└─────────────────────────────────────────────────────────────────┘
```

## Data flow: one training run

```
POST /api/runs {dataset_id, target, task}
  │
  ├─ 1. load Dataset, re-parse its stored CSV text
  │      (the raw text is kept, so a run is always reproducible)
  │
  ├─ 2. build_spec(table, target)
  │      per column: numeric → mean/std/fill
  │                  categorical → category list + __other__ bucket
  │      → FeatureSpec  (serialised into runs.spec_json)
  │
  ├─ 3. transform(table, spec) → X  (n × d float64)
  │      encode_target(table, target, task) → y, classes
  │
  ├─ 4. stratified_split(y, 0.25, seed)
  │      each class keeps its ratio in both halves
  │
  ├─ 5. baseline: majority class / training mean, scored on the holdout
  │
  ├─ 6. for each candidate in build_candidates(task, ...):
  │        fit on train → predict on test → score
  │        (a candidate that raises is recorded with its error, not fatal)
  │
  ├─ 7. sort by primary metric, refit the winner on ALL rows
  │
  └─ 8. persist Run, return RunDetail
```

## Why the encoder is a first-class object

The most common bug in a hand-rolled ML pipeline is training on one encoding
and predicting with another. If you one-hot encode `contract` at training
time and forget the category order, every prediction afterwards is silently
wrong — no error, just bad numbers.

`FeatureSpec` is the fix. It records, per column:

- `kind` — numeric or categorical
- `fill` — the value used for blanks (the column mean)
- `mean`, `std` — for standardisation
- `categories` — the exact ordered list used for one-hot

It is serialised to `runs.spec_json` and replayed by `transform()` at
prediction time. `test_spec_round_trips_through_dict` proves the round trip
produces an identical matrix.

## Why the baseline is not optional

A leaderboard without a baseline is a number without a question. The
baseline is computed on the *same holdout split* with the *same metric*, so
the comparison is apples to apples:

- classification → always predict the most common training class
- regression → always predict the training mean

The report also emits notes when the winner fails to beat it, or when the
top two are within 0.01. Both are cases where a naive tool would present a
confident-looking winner that means nothing.

## Why four model families

Each family fails differently, and the pattern of failure is the insight:

| If this wins | It means |
|---|---|
| Logistic / Ridge | The relationship is linear. Extra capacity is fitting noise. |
| Decision tree | There are thresholds. A line can't express it. |
| Random forest | The signal is real but noisy, and averaging helps. |

That is why the churn dataset is interesting: logistic regression wins, and
that is a *finding*, not a disappointment.

## Scaling notes

**What breaks first: synchronous training.** `POST /api/runs` blocks for the
whole search. At 20k rows and these model sizes it is ~1–2 seconds, which is
acceptable. At 200k rows the random forest alone would take minutes and the
request would time out.

The fix, in order:

1. **Job queue.** `POST /api/runs` writes a `pending` Run and returns 202
   with an id. A worker (RQ, Celery, or a thread pool for a single box)
   picks it up. The UI polls `GET /api/runs/{id}` until `status` flips.
   The `Run.status` column already exists for this — it is currently always
   `completed` or `failed`.
2. **Stream the CSV.** `parse_csv` currently reads the whole file into
   memory. A chunked reader with a running type-inference pass would lift
   the 20k-row cap.
3. **Cache the matrix.** `transform()` is deterministic given a spec, so the
   encoded matrix could be cached per dataset and reused across runs on the
   same target.

**What breaks second: the rate limiter.** It counts per process. Behind N
uvicorn workers the effective limit is `limit × N`. Moving the counters to
Redis is a ~20-line change to `ratelimit.py` — the interface is already a
dependency, so no router changes.

**What breaks third: prediction cost.** `POST /runs/{id}/predict` refits the
winner from the stored dataset on every call. That is deliberate — it
guarantees the model scoring your row is the one the leaderboard describes —
but it is O(training) per prediction. The right fix is to serialise the
fitted model alongside the spec and load it, with a hash of the spec stored
next to it so a mismatch is detectable rather than silent.

## Security posture

- Passwords: bcrypt via passlib, cost 12.
- Tokens: HS256, `sub` = user id only, 12-hour expiry. No role or email in
  the payload, so a stolen token can't enumerate anything.
- Ownership: every dataset and run lookup filters by `owner_id`. A foreign
  id returns **404, not 403** — telling someone a resource exists but isn't
  theirs is itself a leak.
- Uploads: extension check, 5 MB cap, UTF-8 with a Latin-1 fallback, and a
  parser that drops ragged rows rather than padding them.
- Headers: `nosniff`, `DENY` framing, strict referrer policy, a restrictive
  permissions policy, and HSTS in production only.
- Errors: a global handler logs the traceback and returns a generic message.
- Secrets: `SECRET_KEY` has no usable production default — the app raises at
  startup if `ENVIRONMENT=production` and it is unset.
