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

## Repo Plan

- **/data/** → Raw and processed datasets  
- **/src/** → Python modules (config, utils)  
- **/notebooks/** → Jupyter notebooks for each stage  
- **/docs/** → Stakeholder memos, framing slides, project artifacts  
- **/homework/** → All homework contributions, organized by stage (`homework2/`, `homework3/`)  
- **/class_materials/** → Local storage for class materials (never pushed)  

