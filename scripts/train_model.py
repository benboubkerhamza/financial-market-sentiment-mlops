"""
Model Training Script

Train machine learning models to predict stock price movements based on technical indicators.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)


def load_and_prepare_data():
    """Load market data and prepare features"""
    print("="*60)
    print("DATA LOADING")
    print("="*60)
    
    data_path = project_root / 'data' / 'processed' / 'market_processed.csv'
    df = pd.read_csv(data_path)
    
    # Normalize column names
    df.rename(columns={'Date': 'date', 'Ticker': 'ticker', 'Close': 'close'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()} unique")
    
    # Remove rows with missing target (returns)
    df = df.dropna(subset=['returns'])
    print(f"✓ After removing missing returns: {len(df)} records")
    
    return df


def create_target_variables(df):
    """
    Create binary target variable for prediction.
    
    Args:
        df: DataFrame with returns column
        
    Returns:
        DataFrame with added target column
    """
    print("\n" + "="*60)
    print("TARGET VARIABLE CREATION")
    print("="*60)
    
    # Binary classification: up (1) or down (0)
    df['target_binary'] = (df['returns'] > 0).astype(int)
    
    print(f"\n📊 Target Distribution (Binary):")
    print(df['target_binary'].value_counts())
    print(f"  Up: {(df['target_binary']==1).sum()/len(df)*100:.1f}%")
    print(f"  Down: {(df['target_binary']==0).sum()/len(df)*100:.1f}%")
    
    return df


def select_features(df):
    """Select features for training"""
    # Technical indicator features
    feature_cols = [
        'ma_5', 'ma_10', 'ma_20', 'ma_50',
        'ema_12', 'ema_26',
        'rsi',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
        'volatility',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3',
        'volatility_lag_1', 'volatility_lag_2', 'volatility_lag_3',
        'rsi_lag_1', 'rsi_lag_2', 'rsi_lag_3',
        'macd_lag_1', 'macd_lag_2', 'macd_lag_3'
    ]
    
    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    print(f"Total features selected: {len(available_features)}")
    print(f"Features: {', '.join(available_features[:10])}...")
    
    return available_features


def temporal_train_test_split(df, test_size=0.2):
    """
    Split data temporally (important for time series).
    Latest data for testing, earlier data for training.
    """
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*60)
    
    df = df.sort_values('date')
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Train set: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")
    print(f"Split ratio: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    
    return train_df, test_df


def train_classification_models(X_train, y_train, X_test, y_test, model_type='binary'):
    """Train and evaluate classification models"""
    print("\n" + "="*60)
    print(f"CLASSIFICATION MODELS ({model_type.upper()})")
    print("="*60)
    
    # Only use Logistic Regression (best performing model)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        if model_type == 'binary':
            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test)
        else:
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"\n📊 {name} Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall:    {test_recall:.4f}")
        print(f"  Test F1:        {test_f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        print(f"\n📋 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'predictions': y_pred_test
        }
    
    return results




def save_best_models(results_binary, scaler, feature_cols):
    """Save the best binary classification model"""
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best classification model (Logistic Regression)
    best_binary = max(results_binary.items(), key=lambda x: x[1]['test_f1'])
    best_binary_name, best_binary_result = best_binary
    
    binary_path = models_dir / f'classifier_binary_{timestamp}.pkl'
    joblib.dump(best_binary_result['model'], binary_path)
    print(f"✓ Saved binary classifier: {best_binary_name}")
    print(f"  Path: {binary_path}")
    print(f"  Test Accuracy: {best_binary_result['test_acc']:.4f}")
    print(f"  Test F1: {best_binary_result['test_f1']:.4f}")
    
    # Save scaler
    scaler_path = models_dir / f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Saved scaler: {scaler_path}")
    
    # Save feature list
    features_path = models_dir / f'features_{timestamp}.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_cols}, f, indent=2)
    print(f"✓ Saved feature list: {features_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'binary_classifier': {
            'name': best_binary_name,
            'test_accuracy': float(best_binary_result['test_acc']),
            'test_precision': float(best_binary_result['test_precision']),
            'test_recall': float(best_binary_result['test_recall']),
            'test_f1': float(best_binary_result['test_f1']),
            'path': str(binary_path)
        },
        'scaler_path': str(scaler_path),
        'features_path': str(features_path),
        'n_features': len(feature_cols)
    }
    
    metadata_path = models_dir / f'metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    return metadata


def main():
    """Main training pipeline"""
    print("\n")
    print("="*60)
    print("STOCK PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create targets
    df = create_target_variables(df)
    
    # Select features
    feature_cols = select_features(df)
    
    # Remove rows with missing features
    df_clean = df.dropna(subset=feature_cols)
    print(f"\n✓ After removing missing features: {len(df_clean)} records")
    
    # Split data temporally
    train_df, test_df = temporal_train_test_split(df_clean, test_size=0.2)
    
    # Prepare features
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Scale features
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler")
    
    # Train binary classification models
    y_train_binary = train_df['target_binary']
    y_test_binary = test_df['target_binary']
    results_binary = train_classification_models(
        X_train_scaled, y_train_binary, 
        X_test_scaled, y_test_binary, 
        model_type='binary'
    )
    
    # Save best model
    metadata = save_best_models(
        results_binary,
        scaler, feature_cols
    )
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return metadata


if __name__ == "__main__":
    main()
