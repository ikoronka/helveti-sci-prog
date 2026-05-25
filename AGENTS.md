# AGENTS.md — Helveti SCI-PROG House Rent Dashboard

AI coding instructions and project rules for all models. Follow every rule unconditionally unless the user explicitly overrides one in the same session.

---

## Project Overview

The Scientific House Rent Dashboard is a modular, reactive data science application designed to ingest, clean, analyze, and interpret real-world housing data. It automates the entire data science pipeline from raw API extraction to advanced AI-synthesized research conclusions.

**Entry point:** `app.py` (Marimo app)
**Data layer:** `db_handler.py` (Kaggle download → SQLite cache)
**LLM layer:** `llm_service.py` (Google Gemini API)
**Dataset:** `iamsouravbanerjee/house-rent-prediction-dataset` (Kaggle)

**Out of scope for this repo:** The notebook `basic EDA and Advanced EDA.2.ipynb` analyses Swiss train delays and is unrelated to this project. Do not reference it, extend it, or treat it as representative of the project's data or patterns. It should be removed or migrated to a separate repository.

---

## Core Features

1. **Automated Data Pipeline:**
   - Uses `kagglehub` to fetch the House Rent Prediction dataset.
   - Automates SQLite database ingestion with `db_handler.py`.
   - Sanitizes data using regex-based cleaning and robust feature engineering.
2. **Advanced Reactive Dashboard (Marimo):**
   - Built with the Marimo framework for reactive, reproducible notebook-based UIs.
   - Features a sidebar for filtering data by City and BHK.
   - Provides real-time interactive updates of KPIs, charts, and tables.
3. **Statistical & Diagnostic Analysis:**
   - Performs univariate and multivariate outlier detection (IQR, Z-score, Isolation Forest, LOF).
   - Executes rigorous statistical validation (Shapiro-Wilk, Anderson-Darling, Jarque-Bera for normality).
   - Performs automated linear regression and non-parametric tests (Kruskal-Wallis).
   - Analyzes multicollinearity using Variance Inflation Factor (VIF).
4. **AI-Driven Synthesis:**
   - Integrates Google Gemini API (`google-genai` SDK) for intelligent, data-aware report generation.
   - Synthesizes findings into plain-English research conclusions.

---

## Technical Stack

- **Languages/Frameworks:** Python, Marimo
- **Data/Stats:** Pandas, Seaborn, Matplotlib, Scipy, Statsmodels, Scikit-Learn
- **Database:** SQLite
- **AI/LLM:** Google Gemini API (`gemini-flash-lite-latest`, `gemini-1.5-flash-8b`)
- **Dependency Management:** `uv`
- **Data Ingestion:** Kaggle API (via `kagglehub`)

---

## Project Structure

- `app.py`: Main dashboard application entry point containing layout, analysis, and visualization.
- `db_handler.py`: Database management and cleaning logic.
- `llm_service.py`: LLM interaction layer using `google-genai`.
- `requirements.txt`: Project dependencies managed by `uv`.
- `README.md`: User-facing documentation.

---

## Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini model interaction. Never hardcode; load from environment.
- `KAGGLE_CONFIG_DIR`: Optional override for Kaggle credential directory (default `~/.kaggle`).
- Kaggle API credentials must be present at `~/.kaggle/kaggle.json` (or `$KAGGLE_CONFIG_DIR/kaggle.json`).

---

## 1. Environment & Dependencies

### Rules

- **Always pin versions.** Every entry in `requirements.txt` must use `==` (e.g. `pandas==2.2.2`). Never write bare package names.
- **Keep requirements.txt complete.** The following packages are used in `app.py` but were missing from `requirements.txt` — they must always be present and pinned:
  - `scikit-learn`
  - `statsmodels`
  - `numpy`
- **Update requirements.txt in the same change** as any new import. Never commit code that imports a package not listed in `requirements.txt`.
- **Never hardcode API key values or paths.** All secrets are loaded from environment variables. The canonical variable names are:
  - `GOOGLE_API_KEY` — Google Gemini API key
  - `KAGGLE_CONFIG_DIR` — optional override for Kaggle credential directory (default `~/.kaggle`)
- **Provide `.env.example`.** If one does not exist, create it alongside any change that introduces a new env var. Never commit a populated `.env` file.
- **Do not use `conda` or `pipenv`.** The project uses `uv` + `requirements.txt`. All install instructions must reference `uv pip install -r requirements.txt`.

---

## 2. Data Ingestion (`db_handler.py`)

### Rules

- **Pre-flight Kaggle credential check.** Before any `kagglehub.dataset_download()` call, verify that `~/.kaggle/kaggle.json` (or `$KAGGLE_CONFIG_DIR/kaggle.json`) exists. If not, raise a descriptive `EnvironmentError` with setup instructions — never let kagglehub surface a cryptic auth failure.
- **Preserve decimals in regex cleaning.** The Size column cleaner must use `r"[^0-9.]"` (not `r"[^0-9]"`) to avoid destroying fractional values (e.g. `45.5 sqft` must not become `455`).
- **Validate schema immediately after load.** After reading from SQLite or the raw CSV, assert the expected column set and dtypes before returning the DataFrame. Raise a descriptive `ValueError` listing any missing columns. Expected columns (minimum):
  ```
  BHK, Rent, Size, Floor, Area Type, Area Locality, City,
  Furnishing Status, Tenant Preferred, Bathroom, Point of Contact
  ```
- **Assert non-empty.** After load, assert `len(df) > 0`. If the DB is empty, delete it and re-ingest rather than returning an empty frame silently.
- **Make the DB path configurable.** Use `os.environ.get("DB_PATH", "house_rent.db")` — never a bare string literal for the path.
- **Use logging, not print.** All operational messages in `db_handler.py` must use `logging.getLogger(__name__)`. Remove any `print()` calls.

---

## 3. EDA (`app.py`)

### Rules

- **Seed every sklearn estimator with a random component.** Use `random_state=42` on all of:
  - `IsolationForest` ✓ (already set — do not remove)
  - `LocalOutlierFactor` ✗ (missing — always add `random_state=42`)
  - `PowerTransformer` (add `random_state=42` even though sklearn currently ignores it, for forward-compatibility)
- **Define magic numbers as named constants** at the top of the cell or function that uses them. Never inline them in the estimator call. Required constants:
  ```python
  IQR_FACTOR: float = 1.5
  ZSCORE_THRESHOLD: float = 3.0
  IF_CONTAMINATION: float = 0.05
  LOF_MIN_NEIGHBORS: int = 5
  LOF_MAX_NEIGHBORS: int = 20
  RANDOM_STATE: int = 42
  ```
- **Always report sample N with normality tests.** Every normality test output (Shapiro-Wilk, Anderson-Darling, Jarque-Bera) must display the sample size N alongside the statistic and p-value. Add a warning if N < 30 (results unreliable) or N > 5000 (Shapiro-Wilk not valid).
- **Extract repeated patterns into helper functions.** The stat-compute → plot block appears multiple times. Refactor into reusable helpers (e.g. `plot_distribution(series, title)`, `compute_group_stats(df, group_col, value_col)`). Never paste the same 10-line block in two cells.
- **No transform fitting on the full dataset.** Any sklearn `fit()` or `fit_transform()` call (e.g. `PowerTransformer`) must operate only on a designated training split, even in exploratory dashboards. For demo purposes, split 80/20 and note this in a comment.

---

## 4. Modeling

### Rules

- **Contextualise every metric passed to an LLM.** When calling `llm_service.py` with regression results, the payload must include:
  - Active filter values (City, BHK range)
  - Sample size N of the filtered slice
  - Units of the target variable (Indian Rupees / month)
  - The model type (e.g. "simple linear regression via scipy.stats.linregress")
  Never pass raw `r_value`, `slope`, `p_value` without this context.
- **Document every engineered feature.** Any new derived column must have a comment block in the cell that creates it:
  ```python
  # Derived features:
  # Rent_per_sqft = Rent / Size  — normalises price by area for cross-city comparison
  # Bath_per_BHK  = Bathroom / BHK — proxy for luxury level
  ```
- **Train/test split precedes any fitting.** Even in demonstration code, always split before fitting transforms or models. Minimum: `train_test_split(df, test_size=0.2, random_state=42)`.
- **Establish a baseline before adding complexity.** Before any non-linear or ensemble model, implement and document a linear baseline (median prediction or OLS) with its RMSE and MAE.

---

## 5. Reproducibility

### Rules

- **Pin all versions before committing.** Run `pip freeze > requirements.txt` (or `uv pip freeze`) and commit the output. Never commit unpinned requirements.
- **Global random seed at notebook/app top.** The first executable cell in any Marimo app or Jupyter notebook must set:
  ```python
  import numpy as np
  np.random.seed(42)
  ```
- **Data integrity assertion on every load.** After `HouseRentDatabase.get_data()`, assert the expected schema and row count immediately. Do not proceed if the assertion fails.
- **No non-deterministic operations without justification.** If a non-deterministic operation is intentional, add a comment explaining why and document the expected variance.

---

## 6. Code Style

### Rules

- **Type hints on every function.** All parameters and return types must be annotated. Example:
  ```python
  def get_data(db_path: str = "house_rent.db") -> pd.DataFrame:
  ```
- **One-line docstring on every function.** Describe what the function returns, not what it does internally.
- **Use `logging`, never `print` in library code.** `db_handler.py` and `llm_service.py` are library modules — they must use `logging.getLogger(__name__)`. `print()` is only acceptable in the Marimo UI cells of `app.py` where user-facing output is intended.
- **No silent exception swallowing.** Every `except` block must either re-raise, log the exception with `logger.exception(...)`, or return a typed error result. Never `pass` or `return None` silently.
- **No leading-underscore variable names for non-private locals.** In Marimo cells, `_var` signals a private/reactive variable. Do not use this prefix for ordinary local variables — it creates confusion about reactivity scope.
- **No bare `except:`.** Always catch specific exception types (e.g. `except ValueError`, `except requests.HTTPError`).
- **Line length:** 88 characters (Black default). Configure your formatter accordingly.

---

## 7. LLM Integration (`llm_service.py`)

### Rules

- **Validate response before returning.** Check that the model's response text is non-empty. If empty, return the fallback string rather than an empty string.
- **Never surface raw exceptions to the UI.** The `generate_insight()` function (or equivalent) must catch all exceptions, log them with `logger.exception(...)`, and return a human-readable fallback string:
  ```
  "Insight generation is currently unavailable. Please check your API key and try again."
  ```
  The UI must display this string — never a Python traceback.
- **Structured prompt with mandatory context fields.** Every prompt sent to Gemini must include:
  ```
  Dataset: House Rent Prediction Dataset (India, {N} listings)
  Active filters: City={city}, BHK={bhk_min}–{bhk_max}
  Metric: {metric_name} = {metric_value} ({units})
  Task: {task_description}
  ```
- **Log every API call at DEBUG level** (prompt length, model name, latency). Log every failure at ERROR level.
- **No retry logic required for MVP**, but the failure branch must log before returning the fallback.
- **Do not expose the API key in logs.** Never log `os.environ.get("GOOGLE_API_KEY")`.

---

## 8. Testing

### Rules

- **Tests live in `tests/`.** Create the directory if absent. Run with `pytest tests/`.
- **Minimum required tests:**
  - `tests/test_db_handler.py`:
    - `test_get_data_returns_dataframe`: asserts return type is `pd.DataFrame`
    - `test_get_data_has_expected_columns`: asserts all required columns are present
    - `test_get_data_non_empty`: asserts `len(df) > 0`
  - `tests/test_llm_service.py`:
    - `test_generate_insight_failure_returns_string`: mock the Gemini client to raise an exception; assert the function returns a non-empty string (not raises)
- **No mocking the database in integration tests.** Use a real SQLite fixture (tmp_path) so schema and dtype assertions are exercised.
- **Every new helper function gets a unit test.** If you extract a reusable function from a Marimo cell, add a corresponding test.

---

## 9. File & Commit Hygiene

### Rules

- **Do not commit `house_rent.db`.** It is gitignored and must stay that way. Never add a DB file to the index.
- **Do not commit `.env`.** Only `.env.example` (with placeholder values) may be committed.
- **The Swiss transit notebook is out of scope.** Do not modify, extend, or reference `basic EDA and Advanced EDA.2.ipynb` in this project. Recommend migrating it to a separate repository.
- **Marimo session cache (`__marimo__/`) must remain gitignored.** Never add these files to the index.
- **Commit messages follow Conventional Commits:** `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.

---

## Future Enhancements

- **Web App Deployment:** Transition the Marimo notebook into a standalone production-ready web application hosted on cloud platforms.
- **Enhanced LLM Prompts:** Incorporate more contextual metrics (e.g., floor-level premiums, furnishing status impacts) into AI prompts for richer synthesized reports.
- **Interactive Exporting:** Implement one-click exports of cleaned datasets and generated reports in CSV or PDF formats.
- **Model Training:** Expand the dashboard to train and visualize machine learning models (e.g., Random Forest or XGBoost) to predict rent prices.
