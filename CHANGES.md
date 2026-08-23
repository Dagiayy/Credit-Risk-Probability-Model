# Engineering Audit & Upgrade Log

This document records the senior-level audit performed on the
**Credit Risk Probability Model** project and every change made to bring it
from a course-assignment prototype to a production-oriented system. It is
organized as: what existed, what was wrong with it, and what changed.

---

## 1. Original state (as received)

A Xente-transactions-based credit risk scorer built around:

- `src/data_processing.py` — EDA + IQR outlier flagging on the raw CSV.
- `src/feature_eng_process.py` — sklearn-style custom transformers
  (aggregate features merged onto every transaction row, datetime parts,
  categorical imputation + one-hot, numeric imputation + scaling), plus a
  standalone WOE/IV calculator that the README claimed was used for encoding
  but the code never actually applied.
- `src/proxy_labeling.py` — RFM computation + KMeans(k=3) to invent an
  `is_high_risk` label (no real default/repayment data exists yet).
- `src/train.py` — trains LogisticRegression + RandomForest, logs to
  MLflow, registers the RandomForest under `CreditRiskModel`.
- `src/predict.py` / `src/api/main.py` — load a **hard-coded model
  version (2)** from the MLflow registry and predict.
- `Dockerfile` + `docker-compose.yml` (MLflow + API services) and a
  flake8+pytest GitHub Actions workflow.
- No `.git` repository existed at all — the "repo" was an unversioned
  extracted folder.

The repo had a reasonable shape (separation into data processing / feature
engineering / labeling / training / serving) — that structure was kept.
What needed fixing was correctness, leakage, and everything around it that
makes a pipeline actually operable.

---

## 2. Problems found

### P0 — Critical (broken, insecure, or fundamentally incorrect)

1. **Target leakage in the serving API.** `pydantic_models.CustomerData`
   required the caller to supply `FraudResult` — the historical fraud label
   from the source dataset — as a model *input*. For a real loan applicant
   this value doesn't exist yet; for historical rows, handing it to the
   model as a feature leaks the very thing the model would otherwise have
   to predict something correlated with.
2. **Pseudo-replication + leakage from wrong modeling grain.** Aggregate
   customer features (e.g. `total_transaction_amount`, computed over *all*
   of a customer's transactions including ones that would land in the test
   split) were merged onto every transaction row, and the train/test split
   was then done at the transaction row level. The same customer could
   appear on both sides of the split, and every row "knew about" its own
   held-out siblings. See `README.md`'s
   [Why customer-grain, not transaction-grain](README.md#why-customer-grain-not-transaction-grain).
3. **Scaling fit on the full dataset before the split** in the original
   `train.py::preprocess` (`StandardScaler().fit_transform` ran before
   `train_test_split`) — classic train/test leakage, independent of #2.
4. **`docker-compose.yml` never actually served the API.** The
   `credit-risk-api` service's `command: python src/train.py` overrode the
   Dockerfile's `uvicorn` CMD, so `docker-compose up --build` ran a training
   job instead of starting the API.
5. **Hard-coded model version (`MODEL_VERSION = 2`)** duplicated in both
   `predict.py` and `api/main.py`. Promoting a new model required editing
   and redeploying source code in two places.
6. **No git repository.** Nothing was version-controlled; the CI workflow
   in `.github/workflows/` could never have run against this state.
7. **Unpinned dependencies** (`requirements.txt` had bare package names, no
   versions) — not reproducible; a fresh install could resolve incompatible
   versions at any time.

### P1 — High (necessary for professional quality)

8. Feature engineering vs. README mismatch: README claimed WOE encoding via
   `optbinning`; the code actually one-hot encoded and never used the
   computed IV values for anything, and `optbinning` wasn't even in
   `requirements.txt`.
9. No data validation between "CSV on disk" and "dataframe used by every
   downstream stage" — a renamed/retyped column would silently propagate.
10. No structured logging anywhere — only `print()` with emoji, unusable in
    a container's aggregated logs.
11. The API loaded its model via unguarded module-level code (no error
    handling, no health/readiness distinction, so a bad MLflow connection
    crashed the whole process at import time and gave no way for an
    orchestrator to tell "starting" from "broken").
12. Only one test file, testing a `preprocess()` function that itself had
    the leakage bug above; no tests for feature engineering, labeling,
    schema validation, or the API.
13. CI ran on `push` to `main` only (not PRs), had no coverage reporting,
    and never verified the Docker image actually builds.
14. No conditional model promotion — every training run re-registered and
    (implicitly, via the hard-coded version number elsewhere) could become
    "the" serving model regardless of whether it was actually better.
15. No explainability — a raw probability is not sufficient justification
    for a credit decision in a Basel-II-aware context (the project's own
    README cites Basel II).
16. Duplicated, stray `notebooks/notebooks/plots/` directory (an artifact
    of a past run with a relative-path bug) committed alongside the correct
    `notebooks/plots/`.

### P2 — Medium

17. No `.env.example` / centralized configuration — MLflow URI, paths, and
    model name were hard-coded (with duplication) across four files.
18. No `.dockerignore`; the API image would have shipped notebooks, tests,
    and the full git history if `COPY . .` were ever combined with a real
    `.git` checkout.
19. Docker image ran as root, had no health check.

---

## 3. What changed

### Data engineering

- **`src/data_schema.py` (new)** — a `pandera` schema for the raw Xente
  contract (types, `FraudResult ∈ {0,1}`, non-negative `Value`, unique
  `TransactionId`). `data_processing.load_data()` validates against it by
  default and raises a readable `DataValidationError` instead of letting a
  malformed file propagate silently.
- **`src/data_processing.py`** — added `duplicate_report()` (exact-row and
  primary-key duplicate detection — previously absent entirely) and
  `data_quality_report()` (a single consolidated report combining missing
  values, duplicates, and IQR outlier counts, usable both in the notebook
  and as a CI smoke check). Matplotlib now uses the `Agg` backend
  explicitly so plotting never tries to open a GUI window in a container/CI.
- Removed the stray duplicated `notebooks/notebooks/plots/` directory.

### ML correctness (the core of this audit)

- **Customer-grain feature aggregation.** `feature_eng_process.build_customer_features`
  now collapses transactions to one row per `CustomerId` (totals, mean, std,
  min/max transaction amount, transaction count, recency/tenure in days,
  per-customer outlier rates, dominant category/channel/provider, most
  common transaction hour) *before* any imputation/encoding/scaling. This
  is what fixes problems #2 and #3 above simultaneously: the split now
  happens on the customer-level table, so it is leak-free and
  non-replicated by construction.
- **Preprocessing pipeline fit strictly on the training split.**
  `train.py` builds the customer-labeled dataset, splits it, and only then
  calls `feature_pipeline.fit_transform(X_train)` / `.transform(X_test)` —
  the scaler and imputer never see test data.
- **The fitted preprocessing pipeline is persisted and versioned with the
  model** (`joblib`, logged as an MLflow artifact next to the model in the
  same run) so `predict.py` and the API apply the *exact* transform used at
  training time, not a re-derived approximation.
- **`proxy_labeling.py`** now reports silhouette score and per-cluster RFM
  profiles on every run (previously the cluster→risk mapping was a single
  unexplained heuristic line) and uses standardized (not raw-scale) RFM for
  choosing the least-engaged cluster, making small clusters/units not
  dominate the choice.
- **WOE/IV** is now explicitly documented and used as a feature-selection
  *diagnostic* (`calculate_woe_iv`, logged per training run) rather than a
  claimed-but-unimplemented encoding step — closing the README/code
  mismatch without adding a heavyweight `optbinning` dependency the project
  doesn't otherwise need.
- **Baseline model.** `train.py` now trains a `DummyClassifier` first so
  the logistic regression / random forest results are reported against a
  meaningful floor, not in isolation.
- **Class imbalance.** Both real candidate models use `class_weight="balanced"`
  (the proxy label is necessarily imbalanced — most customers aren't in the
  smallest, least-engaged cluster).
- **Conditional promotion.** `model_registry.py` (new) resolves and
  promotes models via an MLflow Registry **alias** (`champion`) instead of
  a hard-coded version number. `train.py` only moves the alias if the new
  model's ROC-AUC is at least as good as the current champion's
  (`PROMOTION_MIN_ROC_AUC_GAIN`, default 0.0) — a regression can't silently
  become the serving model.
- **`reports/training_report.json`** — every training run now writes a
  machine-readable summary (baseline + all candidates' metrics, which one
  was selected, whether it was promoted) as a lightweight model card /
  audit trail.

### AI / explainability

- **`src/explainability.py` (new)** — SHAP-based per-prediction
  explanations (`TreeExplainer` for the tree model, falling back to a
  model-agnostic explainer for others). Exposed via `?explain=true` on
  `/api/v1/predict`, returning the top features (with direction) that
  pushed a given customer's score up or down. This directly serves the
  project's own stated Basel-II explainability requirement, which the
  original code only listed as a "next step" and never implemented.

### API / software engineering

- **`src/api/pydantic_models.py`** — replaced the ~30-field pre-engineered
  `CustomerData` schema with `TransactionRecord`/`CustomerTransactionsRequest`,
  which accept **raw transaction fields** (the same shape as a CSV row) and
  validate that a batch belongs to a single customer. `FraudResult` is not
  a field on this schema — enforced by a regression test
  (`test_transaction_schema_does_not_accept_fraud_result`).
- **`src/api/main.py`** — rewritten around a FastAPI `lifespan` handler
  that loads the champion model once at startup and never crashes the
  process if loading fails (logs the error; `/health` still returns 200,
  `/ready` returns 503 until a model is available — standard
  liveness/readiness separation for a container orchestrator). Added
  request logging middleware, a global exception handler that never leaks
  internals in a 500 response, and a versioned route (`/api/v1/predict`).
  The endpoint now runs the real feature pipeline against raw input instead
  of trusting the caller to have pre-computed features correctly.
- **`src/model_registry.py` (new)** — the single place that knows how to
  resolve "the model that should serve traffic" (by alias, with a
  configurable fallback version), shared by `predict.py` and the API so
  the resolution logic can't drift between the two call sites.
- **`src/config.py` (new)** — `pydantic-settings`-based configuration
  consolidating every path, MLflow URI, model name/alias, and modeling
  constant that used to be hard-coded (with duplication) across four files.
  Overridable via `.env` (`.env.example` added).
- **`src/logging_utils.py` (new)** — structured logging shared by every
  entry point; replaced `print()`/emoji debug statements throughout
  `data_processing.py`, `feature_eng_process.py`, `proxy_labeling.py`,
  `train.py`, `predict.py`, and the API.

### Testing

Expanded from one test file (testing a function that no longer exists in
this form) to a suite covering, on synthetic fixtures (no real dataset or
MLflow server required — `tests/conftest.py`):

- `test_data_processing.py` — outlier flagging, duplicate/missing-value
  reporting, data quality report shape.
- `test_data_schema.py` — schema acceptance and four distinct rejection
  cases (missing column, invalid `FraudResult`, duplicate `TransactionId`).
- `test_feature_engineering.py` — one-row-per-customer invariant,
  transaction-count correctness against the source data, std=0 (not NaN)
  for single-transaction customers, pipeline fit/transform column
  stability, and IV calculation sanity.
- `test_proxy_labeling.py` — RFM shape/invariants, binary label output,
  single-cluster-per-risk-label invariant.
- `test_config.py` — settings defaults and environment-variable overrides.
- `test_api.py` — health/readiness probes (including 503-when-unloaded for
  both `/ready` and `/api/v1/predict`), a realistic predict round-trip
  against a model trained on synthetic data inside the test, a 422 on
  mixed-customer batches, and the `FraudResult`-exclusion regression test.

### Security

- No secrets or credentials were found committed in the repository (checked
  across `src/`).
- `.env.example` added with no real values; `.env` was already git-ignored.
- Docker image now runs as a non-root user (`appuser`), and `.dockerignore`
  (new) keeps `.git`, `tests/`, `notebooks/`, `data/`, and `.env*` out of
  the built image.
- The API's global exception handler returns a generic message on
  unexpected errors and logs the real exception server-side, instead of
  letting FastAPI's default behavior potentially surface internals.

### Performance / scalability

- Feature engineering now does one groupby-aggregation pass per customer
  instead of a per-row merge that (at transaction grain) scaled with the
  dataset's row count multiplied by its column count for no analytical
  benefit — the customer-grain table is also what makes the model itself
  scale with the number of *customers*, not transactions, which is the
  correct scaling axis for a customer-level decision.
- `ColumnTransformer.set_output(transform="pandas")` is used so feature
  names survive the pipeline without a manual bookkeeping pass.

### Deployment

- **`docker-compose.yml`** — fixed the broken override so
  `docker-compose up --build` actually starts the API; added an MLflow
  healthcheck that the API service depends on; added a separate
  `trainer` profile (`docker-compose --profile train run --rm trainer`) so
  training runs as an explicit one-off job instead of being conflated with
  "start the service."
- **`Dockerfile`** — non-root user, `HEALTHCHECK` against `/health`,
  `PYTHONUNBUFFERED`/`PYTHONDONTWRITEBYTECODE`, and only copies `src/`
  (see `.dockerignore`).
- **`.github/workflows/ci.yml`** — now runs on PRs as well as pushes,
  caches pip dependencies, runs flake8 against `tests/` too, runs pytest
  with coverage (uploaded as a build artifact), and added a second job that
  verifies the Docker image actually builds.

### Documentation

- `README.md` rewritten to reflect the actual architecture, explain *why*
  customer-grain modeling and the proxy target exist (not just *what* the
  code does), and give accurate run/test/deploy instructions.
- `data/README.md` (new) documents the expected raw-data location, since
  the dataset itself is intentionally not committed.
- This file.

---

## 4. Remaining issues / not done

- **No real dataset was available in this environment**, so the full
  pipeline (`train.py` end-to-end against real Xente data) could not be
  executed here; correctness was instead verified with a synthetic-data
  test suite covering every stage's invariants (grain, leakage, schema,
  API contract). Run `python -m src.train` against the real CSV before
  relying on any specific metric values.
- **No automated drift detection** — flagged as a next step in the README
  rather than implemented, since it requires a live serving history to be
  meaningful and would be premature to build against synthetic data.
- **No authentication on the API** — out of scope for this pass since
  there was no existing auth to audit and the deployment context (internal
  BNPL underwriting service, presumably behind existing bank
  infrastructure) wasn't specified; flagged rather than guessed at.
- **DVC / formal data versioning** was intentionally not introduced — the
  dataset is a single static file for this pilot and adding a data
  versioning tool would be complexity without a corresponding need per the
  project's "do not overengineer" guidance.

## 5. Final assessment

**Before:** prototype — the API had a target-leakage bug that would have
made it unusable for real underwriting, `docker-compose up` didn't start
the service it was supposed to, and there was no version control at all.

**After:** production-oriented / portfolio-grade for a project of this
scope — leak-free modeling grain, environment-driven configuration, a real
data contract, conditional model promotion via the MLflow registry, an API
that runs the actual training-time feature pipeline against raw input,
SHAP-backed explainability, a test suite that exercises every stage without
requiring the real dataset, and CI/CD that actually verifies the things it
claims to (including that the Docker image builds).
