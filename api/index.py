"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a module-level `app` (or `handler`) in the
file named by `builds[].src`. This is the whole adapter — the FastAPI app is
already an ASGI callable, so nothing else is needed.

Note: SQLite will not work here. Vercel's filesystem is read-only outside
/tmp, and /tmp does not persist between invocations. Set DATABASE_URL to a
hosted Postgres instance.
"""

from app.main import app  # noqa: F401
