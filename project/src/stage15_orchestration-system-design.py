# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import pandas as pd
import argparse, logging, sys, pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

tasks = pd.DataFrame({
    'task': [
        'ingest',
        'clean',
        'feature_engineer',
        'train_model',
        'evaluate',
        'report'
    ],
    'inputs': [
        '/data/raw/*.csv',
        'raw_ingested.csv',
        'clean.csv',
        'features.csv',
        'features.csv + model.pkl',
        'metrics.json + model.pkl'
    ],
    'outputs': [
        'raw_ingested.csv',
        'clean.csv',
        'features.csv',
        'model.pkl',
        'metrics.json',
        'report.md'
    ],
    'idempotent': [True, True, True, True, True, True]
})

print("=== Task Decomposition ===")
print(tasks)



def train_model(input_path: str, output_path: str, log_path: str) -> None:
    logging.info("[train_model] start")

    df = pd.read_csv(input_path)

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    X = df[['income', 'transactions']].values
    y = df['spend'].values

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    logging.info("[train_model] MAE = %.4f", mae)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(output_path, "wb"))
    logging.info("[train_model] model saved at %s", output_path)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"MAE: {mae:.4f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train model task")
    parser.add_argument("--input", required=True, help="Processed feature CSV")
    parser.add_argument("--output", required=True, help="Path to save model.pkl")
    parser.add_argument("--log", required=True, help="Path to save metrics log")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    train_model(args.input, args.output, args.log)


if __name__ == "__main__":
    # Example CLI call (for testing)
    main([
        "--input", "../data/processed/features.csv",
        "--output", "../models/model.pkl",
        "--log", "../reports/metrics.txt"
    ])

from pathlib import Path
import pandas as pd
tasks = pd.DataFrame({
    'task': ['ingest', 'clean', 'train_or_score', 'report'],
    'inputs': ['/data/raw.ext', 'prices_raw.json', 'prices_clean.json', 'model.json'],
    'outputs': ['prices_raw.json', 'prices_clean.json', 'model.json', 'report.txt'],
    'idempotent': [True, True, True, True]
})
tasks

dag = {
    'ingest': [],
    'clean': ['ingest'],
    'feature_engineer': ['clean'],
    'train_model': ['feature_engineer'],
    'evaluate': ['feature_engineer', 'train_model'],
    'report': ['evaluate']
}

print("\n=== DAG Dependencies ===")
print(dag)

logging_plan = pd.DataFrame({
    'task': ['ingest', 'clean', 'feature_engineer', 'train_model', 'evaluate', 'report'],
    'log_messages': [
        'start/end, row counts, source URI',
        'start/end, null rates before/after',
        'new feature stats, drift checks',
        'hyperparams, training MAE',
        'eval metrics (MAE, calibration)',
        'report generated path'
    ],
    'checkpoint_artifact': [
        'raw_ingested.csv',
        'clean.csv',
        'features.csv',
        'model.pkl',
        'metrics.json',
        'report.md'
    ]
})

print("\n=== Logging & Checkpoints ===")
print(logging_plan)

def train_model(input_path: str, output_path: str, log_path: str) -> None:
    logging.info("[train_model] start")

    df = pd.read_csv(input_path)

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    X = df[['income', 'transactions']].values
    y = df['spend'].values

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    logging.info("[train_model] MAE = %.4f", mae)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(output_path, "wb"))
    logging.info("[train_model] model saved at %s", output_path)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"MAE: {mae:.4f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train model task")
    parser.add_argument("--input", required=True, help="Processed feature CSV")
    parser.add_argument("--output", required=True, help="Path to save model.pkl")
    parser.add_argument("--log", required=True, help="Path to save metrics log")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    train_model(args.input, args.output, args.log)


if __name__ == "__main__":
    # Example CLI call (for testing)
    main([
        "--input", "../data/processed/features.csv",
        "--output", "../models/model.pkl",
        "--log", "../reports/metrics.txt"
    ])

import time
def retry(n_tries=3, delay=0.2):
    def wrapper(fn, *args, **kwargs):
        # TODO: implement try/except loop with sleep backoff
        return fn(*args, **kwargs)
    return wrapper