# Stage 15: Orchestration & System Design

## 1 Project Task Decomposition
Our project has 6 main tasks:

| Task            | Inputs                           | Outputs                           | Idempotent | Notes |
|-----------------|----------------------------------|-----------------------------------|------------|-------|
| ingest          | `/data/raw/*.csv`                | `/data/interim/raw_ingested.csv` | Yes        | Pull raw data, combine |
| clean           | `raw_ingested.csv`               | `/data/processed/clean.csv`      | Yes        | Drop duplicates, fill NAs |
| feature_engineer| `clean.csv`                      | `/data/processed/features.csv`   | Yes        | Add ratios, rolling stats |
| train_model     | `features.csv`                   | `/models/model.pkl`              | Yes        | Fit regression, save model |
| evaluate        | `features.csv, model.pkl`        | `/reports/metrics.json`          | Yes        | MAE, drift, calibration |
| report          | `metrics.json, model.pkl`        | `/reports/report.md`             | Yes        | Human-readable summary |

## 2 Dependencies
The tasks form a linear DAG:

ingest → clean → feature_engineer → train_model → evaluate → report

python
复制代码

- `evaluate` depends on both `features.csv` and `model.pkl`.
- Only `report` runs after evaluation.
- Parallelization is possible if we add “monitoring” jobs in future.

## 3 Logging & Checkpoints
- **ingest**: log row counts, source URIs; checkpoint = `raw_ingested.csv`
- **clean**: log null rates before/after; checkpoint = `clean.csv`
- **feature_engineer**: log feature stats, PSI drift; checkpoint = `features.csv`
- **train_model**: log hyperparams, training MAE; checkpoint = `model.pkl`
- **evaluate**: log metrics (MAE, calibration error); checkpoint = `metrics.json`
- **report**: log file path; checkpoint = `report.md`

Logs will be written to `/logs/{task}.log`. Checkpoints will allow reruns from intermediate stages.

## 4 Failure Points & Retry
- Ingest: network or missing file → retry 3x with backoff 10s.
- Clean: schema mismatch → fail fast, notify.
- Train: insufficient rows → log critical, stop pipeline.
- Report: file write error → retry once.

## 5 Right-Sizing Automation
Automate **ingest, clean, feature_engineer, train_model, evaluate** now, since they are deterministic and easily reproducible.  
Keep **report** partly manual for now: auto-generate draft (`report.md`), but allow human analyst to edit before publishing. This balances reproducibility with business context.


