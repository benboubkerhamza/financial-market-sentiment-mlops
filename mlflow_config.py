"""
MLflow Configuration

Centralized configuration for MLflow experiment tracking.
"""

import mlflow
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"

# Experiment configuration
EXPERIMENT_NAME = "stock-price-prediction"

# MLflow tags
DEFAULT_TAGS = {
    "project": "financial-market-sentiment-mlops",
    "team": "data-science",
    "framework": "scikit-learn"
}


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME):
    """
    Initialize MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        experiment_id: ID of the created/existing experiment
    """
    # Set tracking URI (file-based backend)
    tracking_uri = str(MLFLOW_TRACKING_URI.absolute()).replace('\\', '/')
    mlflow.set_tracking_uri(f"file:///{tracking_uri}")
    
    # Create experiment if it doesn't exist
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✓ Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✓ Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def log_model_metrics(metrics: dict, prefix: str = ""):
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for metric names (e.g., "train_", "test_")
    """
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{prefix}{metric_name}", metric_value)


def log_model_params(params: dict):
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter names and values
    """
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def log_dataset_info(dataset_info: dict):
    """
    Log dataset information as MLflow tags.
    
    Args:
        dataset_info: Dictionary with dataset information
    """
    for key, value in dataset_info.items():
        mlflow.set_tag(f"dataset.{key}", value)
