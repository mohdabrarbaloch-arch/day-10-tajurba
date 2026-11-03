"""A small in-process rate limiter, as a FastAPI dependency.

Why not slowapi: its decorator wraps the endpoint in a way that can drop the
function signature, which makes FastAPI treat a Pydantic body as a query
parameter. A dependency keeps the signature intact and is easier to test.

The trade-off is honest and documented: this counts per process. Behind
multiple workers each worker keeps its own window, so the effective limit is
`limit * workers`. For a single-container deployment that is fine; for a
multi-worker one, move the counters to Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

# key -> timestamps of recent hits
_hits: dict[str, deque[float]] = defaultdict(deque)


def _parse(rule: str) -> tuple[int, int]:
    """'10/minute' -> (10, 60)."""
    count_s, _, period = rule.partition("/")
    count = int(count_s)
    seconds = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
    }.get(period.strip().lower(), 60)
    return count, seconds


def rate_limit(rule: str, scope: str):
    """Build a dependency that enforces `rule` per client IP within `scope`."""
    limit, window = _parse(rule)

    def dependency(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        key = f"{scope}:{client}"
        now = time.monotonic()
        bucket = _hits[key]

        while bucket and now - bucket[0] > window:
            bucket.popleft()

        if len(bucket) >= limit:
            retry = int(window - (now - bucket[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Try again in {retry}s.",
                headers={"Retry-After": str(retry)},
            )

        bucket.append(now)

    return dependency


def reset() -> None:
    """Clear all counters. Used by tests so one test cannot rate-limit another."""
    _hits.clear()
