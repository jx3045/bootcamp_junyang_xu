# Bootcamp Project: Financial Data Analysis

**Stage:** Problem Framing & Scoping (Stage 01)

---

## Problem Statement

The goal of this project is to analyze financial market data to understand and manage risks associated with index futures. Specifically, we aim to identify factors that contribute to extreme market movements and provide actionable insights to stakeholders for informed decision-making. Understanding these risks helps improve portfolio management, hedge strategies, and overall financial stability.

---

## Stakeholder & User

- **Primary Stakeholder:** Risk Management Team  
- **End Users:** Portfolio Managers, Traders, and Analysts  
- **Context:** The analysis informs daily trading decisions and strategic hedging actions. Results are reviewed weekly in risk meetings.

---

## Useful Answer & Decision

- **Type:** Descriptive insights on historical risk, predictive models for future market movements  
- **Metrics/Artifacts:** Volatility metrics, Value-at-Risk (VaR), stress-test reports, risk dashboards  

---

## Assumptions & Constraints

- Historical market data is accurate and complete for the last 24 months  
- Analysis focuses only on index futures, not individual stocks  
- Predictions are probabilistic and cannot capture all market shocks  

---

## Known Unknowns / Risks

- Data latency or missing updates may affect calculations  
- Extreme events (black swans) may not be captured  
- Model assumptions may fail under unusual market conditions  

---

## Lifecycle Mapping

| Goal                        | Stage                                | Deliverable                        |
| --------------------------- | ------------------------------------ | ---------------------------------- |
| Understand risk drivers     | Problem Framing & Scoping (Stage 01) | Stakeholder memo / framing slide   |
| Prepare environment & tools | Tooling Setup (Stage 02)             | Configured Python project scaffold |
| Explore and summarize data  | Python Fundamentals (Stage 03)       | Notebooks, summary CSV/plots       |

---

## Repo Plan

- **/data/** → Raw and processed datasets  
- **/src/** → Python modules (config, utils)  
- **/notebooks/** → Jupyter notebooks for each stage  
- **/docs/** → Stakeholder memos, framing slides, project artifacts  
- **/homework/** → All homework contributions, organized by stage (`homework2/`, `homework3/`)  
- **/class_materials/** → Local storage for class materials (never pushed)  



## Stage 02: Tooling Setup

**Goal:** Set up Python project environment and scaffolding for homework and project work.  

**Key Tasks:**

- Create virtual environment (conda)  
- Install required packages (`requirements.txt`)  
- Set up `.gitignore` to exclude `.env`, local data, caches, and checkpoints  
- Organize folders: `/data/`, `/src/`, `/notebooks/`, `/homework/homework2/`  
- Add basic project README.md entries and update commit history  

**Deliverables:**

- `requirements.txt` with necessary packages  
- `.gitignore` with exclusions (`.env`, `data/`, `__pycache__/`, `.ipynb_checkpoints/`)  
- Homework2 notebook and any starter code in `/homework/homework2/`  

---

## Stage 03: Python Fundamentals

**Goal:** Explore and summarize dataset using Python, NumPy, pandas, and reusable functions.  

**Key Tasks:**

1. **NumPy Operations**  
   - Create arrays and perform elementwise operations  
   - Compare loop vs vectorized execution  

2. **Dataset Loading**  
   - Load `data/starter_data.csv` using pandas  
   - Inspect with `.info()` and `.head()`  

3. **Summary Statistics**  
   - Calculate `.describe()` for numeric columns  
   - Perform `.groupby()` aggregation by category  

4. **Save Outputs**  
   - Save summary stats to `data/processed/summary.csv`  
   - **Bonus:** Create and save a basic plot  

5. **Reusable Functions**  
   - Write at least one utility function (e.g., `get_summary_stats(df)`)  
   - Optionally move function to `src/utils.py` and import in notebook  

**Deliverables:**

- Notebook: `notebooks/hw03_python_fundamentals.ipynb`  

- Output summary: `data/processed/summary.csv`  

- Optional: `src/utils.py`  

---

## Stage 04: Data Acquisition & Ingestion

- API: Alpha Vantage (if key present) or Stooq CSV fallback
- Scrape: Wikipedia S&P 500 constituents table (`id="constituents"`)
- Validation: required columns, dtypes, NA counts
- Outputs: saved raw CSV to `data/raw/` with timestamped filenames
- Notebook: `notebooks/stage04_data-acquisition-and-ingestion_homework-starter.ipynb`
- Archived: `homework/homework4/`

## Stage 05: Data Storage

We organize data into two folders:

- `data/raw/` → Raw data saved as **CSV**
- `data/processed/` → Processed data saved as **Parquet**

The locations are driven by environment variables in `.env`:

DATA_DIR_RAW=data/raw
DATA_DIR_PROCESSED=data/processed

### Formats
- **CSV**: human-readable, portable, universally supported
- **Parquet**: columnar storage, efficient for big data, preserves dtypes

### Utilities
- `write_df(df, path)`: save DataFrame as `.csv` or `.parquet`
- `read_df(path)`: load DataFrame, parse dates automatically if present
- Handles missing directories and missing parquet engines gracefully

### Validation
- Reloaded datasets are checked for:
  - shape equality
  
  - `date` column is datetime
  
  - `price` column is numeric
  

##Stage 06:  Cleaning Strategy 

 For preprocessing, we applied the following steps:

 1. **Missing Values**
     - Filled numeric missing values with column median (`fill_missing_median`).
     - Dropped columns with more than 50% missing data (`drop_missing`).
 2. **Normalization**
     - Scaled selected numeric columns (e.g., `price`) to [0,1] range (`normalize_data`).
     - Improves comparability between features.
  
 3. **Reproducibility**
     - All functions are modularized in `src/cleaning.py`.
     - Notebook demonstrates transformations step by step.
     - Cleaned dataset saved to `data/processed/`.

 This ensures datasets are consistent, reproducible, and ready for further analysis.

# Stage 07 — Outliers & Risk Assumptions

## Overview
This assignment explores different methods for detecting and handling outliers, and evaluates their impact on simple statistical summaries and regression models.

## Contents
- **notebooks/stage07_outliers-risk-assumptions_homework.ipynb**  
  Main notebook containing code, functions, sensitivity analysis, and reflection.  
- **data/raw/outliers_homework.csv**  
  Provided dataset (or synthetic fallback if missing).  
- **data/processed/outlier_sensitivity.csv**  
  Output file with sensitivity analysis results.  

## Key Steps
1. Load dataset (provided or synthetic).  
2. Implement reusable functions:  
   - `detect_outliers_iqr()`  
   - `detect_outliers_zscore()`  
   - `winsorize_series()` (optional stretch goal)  
3. Apply methods to at least one numeric column and create outlier flags.  
4. Perform sensitivity analysis:  
   - Compare mean, median, std with vs. without outliers.  
   - Fit regression models and compare slope, R², and MAE.  
5. Reflect on method choice, assumptions, observed impacts, and risks.  

## Deliverables
- Jupyter Notebook (`.ipynb`) with all code and reflection.  
- Processed output table (`outlier_sensitivity.csv`).  
- This README.md file summarizing the assignment.  

## Notes
- Only one detection method (IQR **or** Z-score) needs to be applied.  

- Winsorizing is optional (stretch goal).  

- All random elements use a fixed seed (`np.random.seed(42)`) for reproducibility.

## Stage08 — Exploratory Data Analysis (EDA)

**Objectives:**
- Generate and save a synthetic dataset (stored in `data/raw/sample_data.csv`).
- Load the dataset and perform Exploratory Data Analysis (EDA).

**Contents:**
1. Data generation and saving (with injected missing values and outliers).
2. Numerical summary statistics (mean, std, skewness, kurtosis).
3. Distribution analysis (histograms, KDE plots, boxplots).
4. Relationship analysis (scatter plots: income vs spend, age vs spend).
5. Correlation heatmap.
6. Insights, assumptions, and next-step recommendations.

**How to Run:**
- Open `notebooks/stage08_eda.ipynb`.
  
- Run all cells sequentially.
  
- A synthetic dataset `sample_data.csv` will be created under `data/raw/`.
  
- Plots and insights will be displayed directly within the notebook.
  
## Stage09 — Feature Engineering
  
**Goal:**  
 - Build new features based on insights from EDA (Stage08).  
 - Save feature-enhanced dataset for future modeling.
  
**Contents:**  
  1. Load dataset from `data/processed/stage08_clean.csv`.  
  2. Implement at least 4 engineered features:  
    - **spend_income_ratio**: Normalizes spend by income.  
    - **rolling_spend_mean**: Captures 3-day spending trend.  
    - **age_group**: Bins age into categorical groups.  
       - **region_spend_share**: Compares individual spend relative to region’s total spend.  
       - **rolling_income_var**: Local variance of income as proxy for variability.  
  3. Document rationale for each feature (why it matters, based on EDA).  
  4. Save processed dataset to `data/processed/stage09_features.csv`.
  
 **How to Run:**  
  - Open `notebooks/stage09_feature_engineering.ipynb`.  
  - Run all cells.  
  - New dataset will be saved in `data/processed/stage09_features.csv`.
  
 **Deliverables:**  
  - Notebook file: `notebooks/stage09_feature_engineering.ipynb`  
  - Processed data: `data/processed/stage09_features.csv`  

## Stage 10a — Linear Regression Diagnostics

**Audience:** Analytics team & peer reviewers  
**Purpose:** Evaluate initial imputation, fit a baseline linear regression model, and inspect residual diagnostics.

**Files:**
- `stage10a_results.csv` — cleaned dataset with imputed missing values.
- `images/` — diagnostic figures:
  - `residuals_hist.png` — histogram of residuals.
  - `residuals_by_segment.png` — residuals per segment.

**Notes:**
- Missing values in `x_feature` were imputed using mean/median.
- Residual diagnostics verify model assumptions and highlight segments with unusual errors.

**Reproduce:**
Run `stage10a_notebook.ipynb`. Synthetic data is generated if the original CSV is missing.

---

## Stage 10b — Time Series & Classification

**Audience:** Analytics team & peer reviewers  
**Purpose:** Build lag/rolling features and fit either a time-series forecasting or classification model, with proper train/test split.

**Files:**
- `stage10b_results.csv` — model predictions and residuals.
- `images/` — visualizations:
  - `parametric_vs_bootstrap_ci.png` — comparison of parametric and bootstrap CIs.
  - `scenario_fits.png` — scenario sensitivity fits.

**Notes:**
- Confidence intervals demonstrate uncertainty around predictions.
- Alternative scenarios include mean vs median imputation and dropping missing values.
- Residuals-by-segment plots check subgroup behavior.

**Reproduce:**
Run `stage10b_notebook.ipynb`. Missing or corrupted CSV triggers synthetic data creation.

---

## Stage 11 — Evaluation & Risk Communication

**Audience:** Analytics team & stakeholders reviewing model evaluation  
**Purpose:** Quantify uncertainty via parametric vs bootstrap CIs, run scenario sensitivity, and perform subgroup checks.

**Files:**
- `final_stage11_results.csv` — scenario-wise MAE, slope, intercept metrics.
- `images/` — visualizations:
  - `parametric_vs_bootstrap_ci.png` — CI comparison.
  - `scenario_fits.png` — scenario sensitivity.
  - `residuals_by_segment.png` — subgroup residual diagnostics.

**Notes:**
- All missing values were imputed; mean/median methods documented.
- Bootstrapped MAE provides uncertainty quantification.
- Subgroup analysis highlights segments with high residuals.

**Reproduce:**
Run `stage11_notebook.ipynb`. Synthetic data is generated if the original CSV is missing.

---

## Stage 12 — Final Reporting

**Audience:** Product & Analytics leadership (decision-focused)  
**Purpose:** Generate a polished, stakeholder-ready report summarizing risk, return, assumptions, and sensitivity.

**Files:**
- `final_report.md` — stakeholder summary including executive summary, charts, assumptions, sensitivity, and decision implications.
- `final_results.csv` — scenario metrics (return, volatility, sharpe-like ratio).
- `sensitivity_summary.csv` — deltas vs baseline for alternate scenarios.
- `images/` — exported figures:
  - `risk_return.png`
  - `return_by_scenario.png`
  - `metricA_over_time.png`
  - Additional optional visualizations if created.

**Notes:**
- Missing values handled via median (baseline) or mean (alternate).
- Outliers handled via IQR winsorization (baseline) or 3σ capping (alternate).
- Visuals are polished for stakeholder communication with concise takeaways.

**Reproduce:**
Run `stage12_notebook.ipynb`. If `../data/raw/sample_data.csv` is missing, synthetic fallback data is generated automatically.

---

## General Notes

- All images are saved in `/deliverables/images/` with descriptive filenames.  
- Notebooks regenerate synthetic data if the CSVs are unavailable.  
- Ensure the same random seed (`np.random.seed(...)`) is used to reproduce results exactly.
