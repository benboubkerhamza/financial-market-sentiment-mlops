"""
Evaluation module for financial news sentiment MLOps pipeline.

Loads the trained model and the held-out test split, computes a full
suite of classification metrics, and logs them back to MLflow.
"""

import os
import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "financial-sentiment"

LABEL_NAMES = ["Bearish", "Somewhat-Bearish", "Neutral", "Somewhat-Bullish", "Bullish"]


def load_model(models_dir: str = MODELS_DIR):
    """Load the serialised sklearn pipeline from disk."""
    model_path = os.path.join(models_dir, "sentiment_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)


def load_test_split(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test split not found: {path}. Run preprocessing first.")
    return pd.read_csv(path)


def evaluate(data_dir: str = DATA_DIR, models_dir: str = MODELS_DIR) -> dict:
    """
    Evaluate the trained model on the test set.

    Returns:
        Dictionary of evaluation metrics.
    """
    model = load_model(models_dir)
    test_df = load_test_split(data_dir)

    X_test = test_df["text_clean"].fillna("").tolist()
    y_test = test_df["label"].tolist()

    y_pred = model.predict(X_test)

    num_classes = len(set(y_test))
    # Build label names only for classes actually present in the test set
    present_labels = sorted(set(y_test))
    label_names = [LABEL_NAMES[i] for i in present_labels if i < len(LABEL_NAMES)]

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    for k, v in metrics.items():
        logger.info("%s: %.4f", k, v)

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        report_path = os.path.join(models_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        conf_path = os.path.join(models_dir, "confusion_matrix.json")
        with open(conf_path, "w") as f:
            json.dump(conf_matrix, f, indent=2)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(conf_path)
        logger.info("Evaluation artifacts saved and logged to MLflow.")

    return metrics


if __name__ == "__main__":
    evaluate()
