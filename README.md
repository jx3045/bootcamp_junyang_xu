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
- **/homework/** → All homework contributions, organized by stage  
- **/class_materials/** → Local storage for class materials (never pushed)  

**Update Cadence:** Commit after each stage or significant milestone. Push regularly to GitHub to maintain history and reproducibility.
