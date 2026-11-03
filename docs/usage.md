# Usage

## The five-minute tour

1. **Sign in** with the demo account, or create one.
2. **Load a sample** — click any of the three cards. This uploads the CSV
   through the same endpoint a real file uses, so nothing is faked.
3. **Open the dataset** — you get the columns, an 8-row preview, and a
   suggested target.
4. **Run AutoML** — pick the target and task, hit the button.
5. **Read the leaderboard** — the winner is highlighted, the baseline sits
   at the bottom in italics.
6. **Try a prediction** — fill the source columns, get a label or a number.

## Reading the leaderboard

The table has one row per candidate plus a baseline row. The columns depend
on the task:

**Classification** — `accuracy`, `macro_f1`

Macro F1 is the unweighted mean of per-class F1. It is there because
accuracy alone lets a model ignore a rare class and still look fine. If
accuracy is 0.90 and macro F1 is 0.55, the model is predicting the majority
class and calling it a day.

**Regression** — `r2`, `rmse`, `mae`

R² can go negative. A negative R² means the model is worse than always
predicting the mean, and it is shown as-is rather than clipped to zero.

### The four stat cards

| Card | What it means |
|---|---|
| **primary metric** | The winner's score on the holdout |
| **baseline** | What you have to beat |
| **lift** | `winner − baseline`. Green if positive, red if not |
| **winner** | The model name and family |

If lift is negative, the report says so in a note above the table. That is
not a bug — it means the features carry no signal for that target, and
knowing that is worth more than a fake 0.9.

## Choosing a target

The app suggests one, using three signals in order:

1. A name matching a known convention (`target`, `label`, `class`, `score`,
   `price`, `churned`, …) — substring match, so `final_score` hits.
2. A low-cardinality column: 2–20 distinct values covering less than half
   the rows. That is what a label looks like.
3. The last column, the usual export convention.

The suggestion is a default, not a decision. Change it in the dropdown.

## Choosing a task

- **Classification** — the target is a category. Needs 2–50 distinct values.
  More than 50 is rejected with a hint to try regression.
- **Regression** — the target is a number. Every value must parse as a
  number; one `N/A` is rejected with the offending value named.

## Preparing your CSV

The parser is deliberately strict, and each rule exists because the
alternative produces silently wrong numbers:

| Rule | Why |
|---|---|
| Header row required, unique names | Duplicate columns make the encoder ambiguous |
| At least 20 data rows | Fewer cannot produce an honest train/test split |
| Max 200 columns | Guards against a mis-delimited file |
| Max 20,000 rows | Memory bound; the file is parsed whole |
| Ragged rows dropped | Padding a short row invents data |
| Blank lines skipped | Common in hand-edited exports |
| UTF-8, with Latin-1 fallback | Covers Excel exports from Windows |
| Delimiter sniffed (`,` `;` `\t` `\|`) | European Excel uses semicolons |

Blank cells are allowed. A blank in a numeric column is filled with that
column's mean; a blank in a categorical column becomes its own `__missing__`
category.

## Understanding the encoding

The "What was encoded" panel lists the exact feature names the models saw.
For a numeric column that is just the column name. For a categorical column
it is one entry per category plus an `__other__` bucket:

```
contract=monthly
contract=yearly
contract=two_year
contract=__other__
```

The `__other__` bucket is what makes prediction safe: a value the model
never saw at training time lands there instead of crashing or being
silently dropped.

## Making a prediction

Fill in the source columns (not the target) and submit. The response is:

- **classification** — the predicted class label, plus per-class
  probabilities when the winning model exposes them (logistic regression
  does; trees and forests do not, so `probabilities` is `null`).
- **regression** — a number, rounded to 4 decimals.

The model is refit from the stored dataset on each call. That is slower than
loading a pickle, but it guarantees the model scoring your row is the one
the leaderboard describes.

## Worked example: iris

1. Load the **Iris species** sample.
2. Target `species`, task `classification`, 25% test split.
3. Result: `decision_tree(depth=4)` at **1.0000** accuracy against a
   **0.3333** baseline.
4. The report notes the top four models are tied — prefer the simpler one.
5. Predict with `5.1 / 3.5 / 1.4 / 0.2` → **setosa**.

## Worked example: churn

1. Load the **Customer churn** sample.
2. Target `churned`, task `classification`.
3. Result: `logistic_regression(lr=0.5, l2=0.01)` at **0.7500** against a
   **0.5100** baseline.
4. Note that the *linear* model beat the forest. On this data that is the
   finding: the relationship is linear and the forest's extra capacity is
   fitting noise.

## Common mistakes

**Predicting with the target column filled in.** The predict form only
builds inputs for non-target columns, so this can't happen through the UI.
Through the API, extra keys are ignored.

**Expecting a high score on a small dataset.** A 30-row file with a 25%
split gives 7 test rows. One wrong prediction moves accuracy by 14 points.
The 20-row minimum is a floor, not a recommendation — 200+ rows is where
scores start meaning something.

**Reading accuracy on an imbalanced target.** If 90% of rows are class 0,
a model that always says 0 gets 0.90. Check macro F1 and the baseline row.
