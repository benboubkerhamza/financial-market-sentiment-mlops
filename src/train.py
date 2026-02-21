"""
Model training module for financial news sentiment MLOps pipeline.

Trains a TF-IDF + Logistic Regression baseline model (fast, reproducible)
and logs parameters, metrics, and the model artifact to MLflow.
"""

import os
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "financial-sentiment"

# Hyperparameters (can be overridden via environment variables)
MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "20000"))
NGRAM_MAX = int(os.getenv("TFIDF_NGRAM_MAX", "2"))
C = float(os.getenv("LR_C", "1.0"))
MAX_ITER = int(os.getenv("LR_MAX_ITER", "500"))


def load_split(split: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}. Run preprocessing first.")
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline with TF-IDF vectoriser + LR classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    ngram_range=(1, NGRAM_MAX),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=MAX_ITER,
                    class_weight="balanced",
                    # "saga" supports both L1 and L2 and scales well to large
                    # feature matrices produced by high-dimensional TF-IDF vectors
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(data_dir: str = DATA_DIR, models_dir: str = MODELS_DIR) -> str:
    """
    Train the sentiment model and persist it.

    Returns:
        Path to the saved model artifact.
    """
    train_df = load_split("train", data_dir)
    val_df = load_split("val", data_dir)

    X_train = train_df["text_clean"].fillna("").tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text_clean"].fillna("").tolist()
    y_val = val_df["label"].tolist()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        params = {
            "tfidf_max_features": MAX_FEATURES,
            "tfidf_ngram_max": NGRAM_MAX,
            "lr_C": C,
            "lr_max_iter": MAX_ITER,
        }
        mlflow.log_params(params)

        pipeline = build_pipeline()
        logger.info("Training on %d samples …", len(X_train))
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="weighted")

        mlflow.log_metrics({"val_accuracy": val_acc, "val_f1_weighted": val_f1})
        logger.info("Validation accuracy=%.4f  f1=%.4f", val_acc, val_f1)

        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "sentiment_model.joblib")
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)
        logger.info("Model saved to %s", model_path)

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run_id: %s", run_id)

    return model_path


if __name__ == "__main__":
    train()
