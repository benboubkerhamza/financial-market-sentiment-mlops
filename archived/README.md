# Archived Files

This folder contains code that was developed during the project but is not needed for the production model.

## Why These Files Were Archived

The project experimented with multiple approaches. After evaluation, **Logistic Regression** (83.6% accuracy) proved to be the best performing model. The following components were archived because they either underperformed or are not needed for the production system:

## Archived Components

### 1. Sentiment Analysis Module (`sentiment/`)
- **Reason**: Limited data coverage (only 90 news articles over 2 days vs 5010 market records)
- **Impact**: Sentiment features didn't improve model performance
- **Contains**: VADER and FinBERT sentiment analyzers

### 2. LSTM Model (`train_lstm_model.py`)
- **Reason**: Underperformed baseline models (49.5% accuracy vs 83.6% for Logistic Regression)
- **Root Cause**: Insufficient training data (~5k records, needs 50k+ for deep learning)
- **Approach**: 30-day sliding window with 5 LSTM layers

### 3. Model Comparison Script (`compare_models.py`)
- **Reason**: One-time analysis tool, not needed for production
- **Purpose**: Generated comparison report between baseline and LSTM models

### 4. Feature Combination Scripts
- `apply_sentiment.py`: Applied sentiment analysis to news data
- `combine_features.py`: Merged sentiment with market features
- **Reason**: Not needed since sentiment features were not beneficial

### 5. Empty Placeholder Files
- `train.py`: Empty module placeholder
- `evaluate.py`: Empty module placeholder
- **Reason**: Never implemented, functionality exists elsewhere

## Production Components (Active)

The following scripts remain active for production:
- `scripts/collect_data.py`: Data collection
- `scripts/preprocess_data.py`: Feature engineering
- `scripts/train_model.py`: Logistic Regression training (cleaned)
- `scripts/predict.py`: Production prediction system

## Metrics Comparison

| Model | Accuracy | F1 Score | Use Case |
|-------|----------|----------|----------|
| **Logistic Regression** | **83.6%** | **0.84** | **PRODUCTION** |
| LSTM | 49.5% | 0.46 | Archived |

## Future Considerations

These archived components could be useful if:
- More news data becomes available (10x increase)
- Dataset grows significantly (50k+ records for LSTM)
- Alternative sentiment sources are identified
