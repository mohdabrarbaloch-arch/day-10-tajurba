# Setup

## Requirements

- Python 3.11+
- pip
- (optional) Docker + Docker Compose for the Postgres path

## Local, SQLite, zero config

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/make_samples.py     # writes data/*.csv
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The database file `tajurba.db` is created on
first start.

### Seed a demo account

```bash
python -c "
from app.db import init_db, SessionLocal
from app.models import User
from app.security import hash_password
init_db()
db = SessionLocal()
db.add(User(email='demo@tajurba.pk', name='Demo Analyst',
            password_hash=hash_password('tajurba-demo-2026')))
db.commit()
print('seeded')
"
```

Then sign in with `demo@tajurba.pk` / `tajurba-demo-2026`.

## Docker + PostgreSQL

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SECRET_KEY
docker compose up --build
```

The app waits for Postgres to pass its healthcheck before starting. Data
lives in the `pgdata` volume.

> **Note:** `docker-compose.yml` uses `postgresql+psycopg://`, which needs
> `psycopg[binary]`. If you hit `ModuleNotFoundError: psycopg`, add
> `psycopg[binary]==3.2.3` to `requirements.txt` and rebuild. The SQLite path
> needs nothing extra.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` enforces a real `SECRET_KEY` |
| `SECRET_KEY` | *(empty)* | **Required in production.** App raises at startup without it |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `720` | 12 hours |
| `DATABASE_URL` | `sqlite:///./tajurba.db` | Any SQLAlchemy URL |
| `MAX_UPLOAD_BYTES` | `5242880` | 5 MB |
| `MAX_ROWS` | `20000` | Rows parsed per file |
| `LOGIN_RATE_LIMIT` | `10/minute` | Applied to `POST /api/auth/login` |
| `TRAIN_RATE_LIMIT` | `20/hour` | Declared; see the note below |
| `CORS_ORIGINS` | `*` | Comma-separated, or `*` |
| `DEBUG` | `false` | Verbose logging |

### Honest note on `TRAIN_RATE_LIMIT`

`TRAIN_RATE_LIMIT` is read into settings but **is not currently enforced** on
`POST /api/runs`. Only register (hardcoded 5/minute) and login (from
`LOGIN_RATE_LIMIT`) are rate limited. The setting is kept because the
dependency is already written — wiring it up is one `dependencies=[...]`
argument on the route. It is listed here rather than quietly left out.

## Running the tests

```bash
pytest -q
ruff check app tests scripts
black --check app tests scripts
```

Expected: `69 passed`, ruff clean, black clean.

Tests use a temporary SQLite file per session (set in `tests/conftest.py`),
so they never touch your dev database.

## Smoke test against a running server

```bash
uvicorn app.main:app --port 8021 &
python scripts/smoke.py http://127.0.0.1:8021
```

This prints the exact leaderboard numbers the README quotes, so you can
verify them rather than trust them. Expected: `18 passed, 0 failed`.

## Screenshots

```bash
pip install playwright
playwright install chromium
uvicorn app.main:app --port 8021 &
python scripts/shoot.py
```

Writes six PNGs to `screenshots/` and fails loudly on any console error or
mobile horizontal overflow.

## Deploying to Vercel

There is a `vercel.json` in this repo, plus `api/index.py` as the ASGI
entrypoint. Two steps:

```bash
npm i -g vercel
vercel --prod
```

Set these in the Vercel dashboard (Project → Settings → Environment
Variables):

- `SECRET_KEY` — a real 48-byte token
- `ENVIRONMENT` — `production`
- `DATABASE_URL` — a hosted Postgres URL (Neon, Supabase, RDS). **SQLite will
  not work on Vercel** — the filesystem is read-only outside `/tmp`, and
  `/tmp` does not persist between invocations.

Then redeploy so the variables take effect.

## Troubleshooting

**`error reading bcrypt version`** — a harmless warning from passlib probing
an older bcrypt API. Hashing works. Pinning `bcrypt==4.2.1` (as
`requirements.txt` does) keeps it quiet.

**`SECRET_KEY must be set when ENVIRONMENT=production`** — exactly what it
says. Generate one and set it.

**Upload rejected with "At least 20 data rows"** — the parser refuses files
too small to split honestly. This is intentional; a 5-row file cannot
produce a meaningful train/test split.

**A column you expected to be numeric is being one-hot encoded** — one
non-blank value in that column doesn't parse as a number. Check for `N/A`,
`-`, or a stray header row.
