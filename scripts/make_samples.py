"""Generate the three sample datasets the app ships with.

These are synthetic but not arbitrary: each one has a known signal, so a
model that fails to find it is genuinely broken rather than just unlucky
with real-world noise. The churn set deliberately includes a weak signal
and a class imbalance, because that is what real churn data looks like.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"


def write(name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows, {len(header)} columns")


def iris() -> None:
    """A compact 3-class set with two clearly separable groups."""
    rng = random.Random(7)
    rows = []
    specs = [
        ("setosa", 5.0, 3.4, 1.5, 0.25),
        ("versicolor", 5.9, 2.8, 4.3, 1.3),
        ("virginica", 6.6, 3.0, 5.6, 2.0),
    ]
    for label, sl, sw, pl, pw in specs:
        for _ in range(50):
            rows.append(
                [
                    label,
                    round(rng.gauss(sl, 0.35), 2),
                    round(rng.gauss(sw, 0.30), 2),
                    round(rng.gauss(pl, 0.40), 2),
                    round(rng.gauss(pw, 0.20), 2),
                ]
            )
    rng.shuffle(rows)
    write(
        "iris.csv",
        ["species", "sepal_length", "sepal_width", "petal_length", "petal_width"],
        rows,
    )


def churn() -> None:
    """Binary churn with a real but modest signal and a 63/37 imbalance.

    The signal is intentionally weak. A model that reports 0.95 here would
    be suspicious, not impressive.
    """
    rng = random.Random(11)
    rows = []
    for _ in range(400):
        tenure = rng.randint(1, 60)
        monthly = round(rng.uniform(500, 4500), 0)
        support_calls = rng.randint(0, 9)
        contract = rng.choice(["monthly", "yearly", "two_year"])
        city = rng.choice(["Karachi", "Lahore", "Islamabad", "Faisalabad", "Multan"])

        # Logit built from the features, then sampled — so the label is
        # genuinely noisy rather than a deterministic function.
        logit = -0.9
        logit += -0.035 * tenure
        logit += 0.00022 * monthly
        logit += 0.28 * support_calls
        logit += {"monthly": 0.85, "yearly": 0.0, "two_year": -0.75}[contract]
        logit += {
            "Karachi": 0.1,
            "Lahore": 0.0,
            "Islamabad": -0.1,
            "Faisalabad": 0.05,
            "Multan": 0.0,
        }[city]
        p = 1 / (1 + 2.718281828**-logit)
        churned = "yes" if rng.random() < p else "no"

        rows.append([churned, tenure, int(monthly), support_calls, contract, city])
    write(
        "churn.csv",
        [
            "churned",
            "tenure_months",
            "monthly_bill",
            "support_calls",
            "contract",
            "city",
        ],
        rows,
    )


def exam_scores() -> None:
    """Regression with a mostly-linear relationship plus mild curvature."""
    rng = random.Random(23)
    rows = []
    for _ in range(300):
        hours = round(rng.uniform(0, 12), 1)
        attendance = rng.randint(40, 100)
        prior = rng.randint(30, 95)
        sleep = round(rng.uniform(4, 9), 1)

        score = (
            18
            + 3.1 * hours
            + 0.28 * attendance
            + 0.22 * prior
            + 1.4 * sleep
            - 0.09 * hours**2
            + rng.gauss(0, 4.5)
        )
        score = max(0.0, min(100.0, score))
        rows.append([round(score, 1), hours, attendance, prior, sleep])
    write(
        "exam_scores.csv",
        ["final_score", "study_hours", "attendance_pct", "prior_score", "sleep_hours"],
        rows,
    )


if __name__ == "__main__":
    iris()
    churn()
    exam_scores()
