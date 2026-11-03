"""CSV parsing with type sniffing, kept deliberately small.

The platform accepts a CSV upload and nothing else. That constraint is
intentional: one format, parsed one way, means the numbers on the
leaderboard can always be traced back to a specific set of rows.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

MAX_ROWS = 20_000
MAX_COLUMNS = 200


class CsvError(ValueError):
    """Raised when an upload cannot be read as a usable table."""


@dataclass
class ParsedTable:
    columns: list[str]
    rows: list[dict[str, str]]
    truncated: bool


def _sniff_delimiter(sample: str) -> str:
    """Pick the delimiter that yields the most columns on the first line."""
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    best, best_count = ",", 0
    for candidate in (",", ";", "\t", "|"):
        count = first_line.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    return best


def parse_csv(text: str, max_rows: int = MAX_ROWS) -> ParsedTable:
    """Parse CSV text into a header list and a list of string-keyed rows.

    Blank lines are skipped. A UTF-8 BOM is stripped. Rows with a different
    column count than the header are dropped rather than padded, because a
    silently padded row is a silently wrong row.
    """
    if not text or not text.strip():
        raise CsvError("The file is empty.")

    text = text.lstrip("\ufeff")
    delimiter = _sniff_delimiter(text)

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration as exc:  # pragma: no cover - guarded by the check above
        raise CsvError("The file has no header row.") from exc

    columns = [h.strip() for h in header]
    if not columns or all(not c for c in columns):
        raise CsvError("The header row is empty.")
    if len(columns) > MAX_COLUMNS:
        raise CsvError(
            f"Too many columns ({len(columns)}); the limit is {MAX_COLUMNS}."
        )
    if len(set(columns)) != len(columns):
        raise CsvError("The header row contains duplicate column names.")

    rows: list[dict[str, str]] = []
    truncated = False
    for raw in reader:
        if not raw or all(not cell.strip() for cell in raw):
            continue
        if len(raw) != len(columns):
            continue
        if len(rows) >= max_rows:
            truncated = True
            break
        rows.append({col: (raw[i] or "").strip() for i, col in enumerate(columns)})

    if not rows:
        raise CsvError("The file has a header but no data rows.")
    if len(rows) < 20:
        raise CsvError(
            f"Only {len(rows)} data rows found. At least 20 are needed to "
            "train and evaluate anything honestly."
        )

    return ParsedTable(columns=columns, rows=rows, truncated=truncated)
