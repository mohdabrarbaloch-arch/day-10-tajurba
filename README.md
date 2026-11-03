# Tajurba

**AutoML that shows its work.**

![Python](https://img.shields.io/badge/python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?style=flat-square&logo=numpy&logoColor=white)
![Tests](https://img.shields.io/badge/tests-69%20passing-2f6b4f?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-b4471f?style=flat-square)

Upload a CSV, pick a column to predict, and Tajurba trains six to eight
candidate models and ranks them. The part that makes it different: **every
score is printed next to a baseline**, and every model is written in plain
NumPy, so you can read the code that produced the number.

---

## Why this exists

Most AutoML tools hand you a leaderboard and ask you to trust it. A 0.94
accuracy looks great until you learn the majority class already gets 0.93.
Tajurba puts the baseline in the same table, on the same metric, so the
question "is this actually good?" has an answer you can check.

The second decision is that nothing is a black box. There is no
scikit-learn, no XGBoost, no autograd. Four model families, each a few
hundred lines you can read:

| Family | What it is | Why it's in the search |
|---|---|---|
| Logistic regression | Multinomial, full-batch GD, L2 on weights only | If this wins, the signal is linear and you don't need anything fancier |
| Ridge regression | Closed-form `(X'X + αI)⁻¹X'y`, centred intercept | The honest baseline for regression |
| Decision tree | CART, gini for classification, variance for regression | Catches thresholds and step functions a line can't |
| Random forest | Bagged trees with per-split feature subsampling | Reduces the variance a single tree has |

A linear model winning tells you something. A forest winning tells you
something else. That contrast is the product.

---

## Screenshots

| Sign in | Datasets |
|---|---|
| ![Sign in](screenshots/01-signin.jpg) | ![Datasets](screenshots/02-datasets.jpg) |

| Dataset preview + train | Leaderboard |
|---|---|
| ![Dataset](screenshots/03-dataset.jpg) | ![Leaderboard](screenshots/04-leaderboard.jpg) |

| Prediction | Mobile |
|---|---|
| ![Prediction](screenshots/05-prediction.jpg) | ![Mobile](screenshots/06-mobile.jpg) |

---

## What it actually scores

These are real runs against the three bundled datasets, not marketing
numbers. Reproduce them with `scripts/smoke.py`.

**Iris — 150 rows, 3 classes, 4 features**

| Model | Accuracy | Macro F1 |
|---|---|---|
| decision_tree(depth=4) | **1.0000** | 1.0000 |
| random_forest(trees=12, depth=6) | 1.0000 | 1.0000 |
| logistic_regression(lr=0.5, l2=0.01) | 0.9444 | 0.9445 |
| logistic_regression(lr=0.3, l2=0.1) | 0.9167 | 0.9165 |
| *baseline (always predict majority)* | *0.3333* | *0.1667* |

Iris is easy and the report says so — it flags that the top four models are
tied and suggests preferring the simpler one.

**Exam scores — 300 rows, regression**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| ridge(alpha=10.0) | **0.6933** | 5.08 | 4.12 |
| random_forest(trees=24, depth=10) | 0.6895 | 5.11 | 4.15 |
| ridge(alpha=1.0) | 0.6884 | 5.12 | 4.14 |
| decision_tree(depth=8) | 0.3543 | 7.37 | 5.87 |
| *baseline (always predict the mean)* | *-0.028* | *7.55* | *—* |

Ridge wins by a hair over the forest, which is the right answer: the target
is mostly linear with mild curvature, so the simpler model should be enough.
The depth-8 tree overfits and lands well below the depth-4 one.

**Customer churn — 400 rows, binary, deliberately weak signal**

| Model | Accuracy | Macro F1 |
|---|---|---|
| logistic_regression(lr=0.5, l2=0.01) | **0.7500** | 0.7500 |
| logistic_regression(lr=0.3, l2=0.1) | 0.7200 | 0.7196 |
| random_forest(trees=12, depth=6) | 0.6900 | 0.6784 |
| decision_tree(depth=8) | 0.6400 | 0.6399 |
| *baseline (always predict majority)* | *0.5100* | *0.3377* |

The churn set is built with a real but modest logit, so the signal is weak
by design. Logistic regression wins here — which is itself the finding: when
a linear model beats the forest, the relationship is linear and the extra
capacity is just fitting noise. A tool that reported 0.95 on this data would
be lying to you.

---

## Features

- **CSV upload with real validation** — delimiter sniffing, BOM stripping,
  ragged-row rejection, a 20-row minimum, and a 5 MB cap. A file that can't
  be trained on is rejected with a reason, not a stack trace.
- **Type inference that refuses to guess** — a column is numeric only if
  *every* non-blank value parses. One `N/A` in a price column makes it
  categorical, because inventing a number is worse than one-hot encoding.
- **A saved encoder** — the `FeatureSpec` (categories, means, stds, fill
  values) is stored with the run, so scoring a new row replays the exact
  transform the model was trained on.
- **Stratified splitting** — each class keeps its proportion in both halves,
  so a rare class can't vanish into the training set.
- **Baseline on every run** — majority class for classification, training
  mean for regression, scored on the same holdout.
- **Honest notes** — the report tells you when the winner didn't beat the
  baseline, and when the top two are within 0.01 of each other.
- **Per-class F1 and macro F1** — a model that ignores a rare class can't
  hide behind accuracy.
- **JWT auth with bcrypt**, per-user data isolation (a foreign dataset
  returns 404, not 403 — existence itself is a leak).
- **Rate limiting** on register and login, security headers on every
  response, and a global handler that never leaks a traceback.
- **Mobile-first UI** — verified at 390px with zero horizontal overflow.

---

## Quick start

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-10-tajurba.git
cd day-10-tajurba

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Generate the sample datasets
python scripts/make_samples.py

# Optional: seed a demo account
python -c "
from app.db import init_db, SessionLocal
from app.models import User
from app.security import hash_password
init_db()
db = SessionLocal()
db.add(User(email='demo@tajurba.pk', name='Demo Analyst',
            password_hash=hash_password('tajurba-demo-2026')))
db.commit()
"

uvicorn app.main:app --reload
```

Open <http://localhost:8000>. API docs at `/api/docs`.

**Demo account:** `demo@tajurba.pk` / `tajurba-demo-2026`

### With Docker + Postgres

```bash
cp .env.example .env
# Set SECRET_KEY in .env:
python -c "import secrets; print(secrets.token_urlsafe(48))"

docker compose up --build
```

---

## Live demo

**Not deployed.** There is no Vercel token in the environment this was built
in, so there is no live URL and I'm not going to invent one. The repo is
deploy-ready: `Dockerfile`, `docker-compose.yml` with Postgres 16, and a
`vercel.json` + `api/index.py` serverless entrypoint. See
[docs/setup.md](docs/setup.md) for the deploy steps.

---

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/health` | Liveness + version |
| `POST` | `/api/auth/register` | Create an account, returns a JWT |
| `POST` | `/api/auth/login` | Sign in |
| `GET` | `/api/auth/me` | Current user |
| `POST` | `/api/datasets` | Upload a CSV (multipart) |
| `GET` | `/api/datasets` | List your datasets |
| `GET` | `/api/datasets/{id}` | Columns, 8-row preview, suggested target |
| `DELETE` | `/api/datasets/{id}` | Delete a dataset and its runs |
| `POST` | `/api/runs` | Train every candidate, return the leaderboard |
| `GET` | `/api/runs` | List your runs |
| `GET` | `/api/runs/{id}` | Full leaderboard, notes, feature names |
| `DELETE` | `/api/runs/{id}` | Delete a run |
| `POST` | `/api/runs/{id}/predict` | Score one row against the winner |
| `GET` | `/api/samples` | The three bundled datasets |
| `GET` | `/api/samples/{key}` | Download one as CSV |

Full request/response shapes: [docs/api.md](docs/api.md).

---

## Tests

```bash
pytest -q          # 69 passed
ruff check app tests scripts
black --check app tests scripts
```

The suite covers metric arithmetic against hand-computed values, model
behaviour (ridge recovers known coefficients, a depth-1 tree can't fit a
4-step function, a forest beats a single tree on noisy data), the encoder's
round-trip through JSON, stratified splitting, and a full
register → upload → train → predict flow through the real FastAPI app.

---

## Known limits

Stated plainly, because a README that only lists strengths is a sales page:

- **Training is synchronous.** `POST /api/runs` blocks until every candidate
  finishes. Fine for 20k rows and small models; wrong for anything bigger.
  The fix is a job queue and a polling endpoint.
- **The rate limiter is per-process.** Behind multiple workers the effective
  limit is `limit × workers`. Move the counters to Redis for real
  multi-worker deployments.
- **Prediction refits the winner** from the stored dataset rather than
  loading a serialised model. Slower, but it guarantees the model scoring
  your row is the one the leaderboard describes.
- **No cross-validation.** One stratified holdout split. A single split can
  be lucky; k-fold would be the next honest improvement.
- **`MAX_ROWS` is enforced at parse time**, not by streaming — a 20k-row cap
  on a 5 MB file is fine, but it is not a streaming parser.

---

## Stack

Python 3.11 · FastAPI 0.115 · SQLAlchemy 2.0 · Pydantic v2 · NumPy 2.2 ·
python-jose (JWT) · passlib + bcrypt · vanilla JS SPA (no build step) ·
SQLite (dev) / PostgreSQL 16 (compose) · Docker

---

## License

MIT — see [LICENSE](LICENSE).

---

*Day 10 of the Autonomous AI Software Engineer 30-day challenge.*
