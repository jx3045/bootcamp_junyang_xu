# Lifecycle Framework Guide

This guide maps each stage of the ML lifecycle to the corresponding work and code in this repository.

---

## Stage 1: Problem Definition
- **Summary**: Defined the business problem and translated it into a machine learning task.  
  Clarified project objectives, success metrics, and key stakeholders.  
  Outcome: a clear problem framing document guiding the rest of the pipeline.

---

## Stage 2: Project Configuration & Environment Setup
- **Summary**: Established the project structure with `/data/`, `/src/`, `/notebooks/`, and `/reports/` directories.  
  Configured Python environment, installed dependencies, and set up version control with Git.  
  Outcome: a reproducible, collaborative-ready project environment.

---

## Stage 3: Python Fundamentals
- **File**: `stage03_python-fundamentals.py`  
- **Summary**: Basic Python practice and foundational coding structures. Ensured comfort with loops, conditionals, and functions to prepare for later stages.

---

## Stage 4: Data Acquisition and Ingestion
- **File**: `stage04_data-acquisition-and-ingestion_homework-starter.py`  
- **Summary**: Connected to raw data sources, ingested datasets into `/data/raw/`, and verified integrity. Established reproducibility in data ingestion.

---

## Stage 5: Data Storage
- **File**: `stage05_data-storage_homework-starter.py`  
- **Summary**: Designed structured storage for raw and processed data. Introduced version control for datasets to enable reproducibility.

---

## Stage 6: Data Preprocessing
- **File**: `stage06_data-preprocessing_homework-starter.py`  
- **Summary**: Cleaned data, handled missing values, standardized formats, and prepared `/data/processed/` outputs for modeling.

---

## Stage 7: Outliers, Risk, and Assumptions
- **File**: `stage07_outliers-risk-assumptions_homework-starter.py`  
- **Summary**: Analyzed data distribution, detected outliers, and clarified assumptions. Ensured robustness of downstream modeling.

---

## Stage 8: Exploratory Data Analysis (EDA)
- **File**: `stage08_exploratory-data-analysis_homework-starter.py`  
- **Summary**: Conducted EDA to identify key trends and correlations. Generated visualizations to inform feature engineering.

---

## Stage 9: Feature Engineering
- **File**: `stage09_feature-engineering_homework-starter.py`  
- **Summary**: Created new features, such as ratios and rolling statistics, to improve model performance.

---

## Stage 10a: Modeling — Linear Regression
- **File**: `stage10a_modeling-linear-regression.py`  
- **Summary**: Built baseline regression models to understand relationships between predictors and outcomes.

---

## Stage 10b: Modeling — Time Series and Classification
- **Files**:  
  - `stage10b_modeling-time-series-and-classification_homework-starter.py`  
  - `stage10b_modeling_junyangxu.py` (custom implementation)  
- **Summary**: Applied time series forecasting and classification techniques. Extended modeling scope beyond regression.

---

## Stage 11: Evaluation & Risk Communication
- **File**: `stage11_evaluation-risk-communication.py`  
- **Summary**: Evaluated models with quantitative metrics and documented risks/limitations. Communicated results effectively.

---

## Stage 12: Results Reporting & Delivery Design
- **File**: `stage12-results-reporting-delivery-design-stakeholder-communication.py`  
- **Summary**: Designed reporting structure for stakeholders. Delivered results with clarity and actionable insights.

---

## Stage 13: Productization
- **File**: `stage13_productization.py`  
- **Summary**: Packaged the pipeline into deployable modules. Added CLI functionality and reusable scripts for repeatable runs.

---

## Stage 14: Deployment and Monitoring
- **Files**:  
  - `stage14_deployment-and-monitoring.py`  
  - `stage14_deployment-and-monitoring/` (supporting folder)  
- **Summary**: Outlined strategies for deploying models into production and monitoring performance over time.

---

## Stage 15: Orchestration & System Design
- **Files**:  
  - `stage15_orchestration-system-design.py`  
  - `stage15_orchestration-system-design/` (supporting folder)  
- **Summary**: Designed orchestration workflows (task DAGs, dependencies, retries). Ensured maintainability of system pipelines.

---

## Stage 16: Lifecycle Review
- **File**: `stage16_lifecycle-review.py`  
- **Summary**: Final polish of repo structure, documentation, and lifecycle mapping. Completed `framework_guide.md` and summary reports.

---
