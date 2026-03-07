# MLflow Integration Guide

## Overview

MLflow experiment tracking has been integrated into the training pipeline to track all experiments, parameters, metrics, and model artifacts.

## What's Tracked

### Parameters
- Model hyperparameters (max_iter, random_state, solver)
- Data split configuration (train_size, test_size, split_ratio)
- Feature engineering (n_features, scaler type)

### Metrics
- **Test metrics**: accuracy, precision, recall, F1-score
- **Train metrics**: accuracy
- **Confusion matrix**: True Positives, False Positives, True Negatives, False Negatives

### Artifacts
- **Model**: Trained Logistic Regression model (sklearn format)
- **Scaler**: StandardScaler for feature preprocessing
- **Features**: JSON file with list of 27 features
- **Metadata**: Complete model metadata

### Tags
- `model_type`: logistic_regression
- `task`: binary_classification
- `target`: price_direction
- `dataset.*`: Dataset information (n_records, n_tickers, date_range)

## Usage

### 1. Train Model with MLflow Tracking

```bash
python scripts/train_model.py
```

All metrics, parameters, and artifacts are automatically logged to MLflow.

### 2. View Experiments in MLflow UI

**Option A: Using the launcher script**
```bash
python scripts/start_mlflow_ui.py
```

**Option B: Direct mlflow command**
```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Then open your browser to: **http://localhost:5000**

### 3. Browse Experiments

In the MLflow UI you can:
- View all experiment runs
- Compare multiple runs side-by-side
- Sort and filter by metrics
- Download model artifacts
- View detailed run information

## Directory Structure

```
financial-market-sentiment-mlops/
├── mlruns/                     # MLflow tracking data
│   └── {experiment_id}/
│       └── {run_id}/
│           ├── metrics/        # Logged metrics
│           ├── params/         # Logged parameters
│           ├── tags/           # Tags and metadata
│           └── artifacts/      # Model artifacts
├── mlflow_config.py            # MLflow configuration
└── scripts/
    ├── train_model.py          # Training with MLflow integration
    └── start_mlflow_ui.py      # MLflow UI launcher
```

## Configuration

The MLflow configuration is centralized in `mlflow_config.py`:

- **Tracking URI**: `file:///mlruns` (local file-based backend)
- **Experiment Name**: `stock-price-prediction`
- **Artifacts**: Stored within mlruns directory

## Experiment Comparison

To compare multiple training runs:

1. Start MLflow UI
2. Select multiple runs using checkboxes
3. Click "Compare" button
4. View side-by-side comparison of:
   - Parameters (hyperparameters, data config)
   - Metrics (accuracy, F1, precision, recall)
   - Artifacts (models, scalers)

## Model Registry (Future)

Current setup uses file-based tracking. For production, consider:
- **SQLite backend**: `sqlite:///mlflow.db`
- **PostgreSQL backend**: For team collaboration
- **Remote tracking server**: For centralized tracking

## Best Practices

### When to Log a New Run
- After changing hyperparameters
- After modifying feature engineering
- After collecting new data
- After changing model architecture

### What to Compare
- Different model architectures (Logistic vs Random Forest)
- Different hyperparameter values
- Different feature sets
- Different training data splits

### Experiment Naming
Runs are automatically named with timestamp:
- Format: `logistic_regression_YYYYMMDD_HHMMSS`
- Example: `logistic_regression_20260308_004536`

## Troubleshooting

### MLflow UI not starting
```bash
# Check if MLflow is installed
pip list | grep mlflow

# Verify mlruns directory exists
ls mlruns/
```

### Port already in use
```bash
# Change port in start_mlflow_ui.py
--port 5001  # instead of 5000
```

### Artifacts not logging
- Ensure models are saved before logging
- Check file paths are correct
- Verify write permissions in project directory

## Next Steps

- [ ] Set up remote MLflow tracking server
- [ ] Implement model registry
- [ ] Add A/B testing comparison
- [ ] Integrate with CI/CD pipeline
