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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")


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


def create_target_variables(df, threshold=0.001):
    """
    Create target variables for prediction.
    
    Args:
        df: DataFrame with returns column
        threshold: Threshold for classifying price movement (default: 0.1% = 0.001)
        
    Returns:
        DataFrame with added target columns
    """
    print("\n" + "="*60)
    print("TARGET VARIABLE CREATION")
    print("="*60)
    
    # Binary classification: up (1) or down (0)
    df['target_binary'] = (df['returns'] > 0).astype(int)
    
    # Multi-class classification: up (2), neutral (1), down (0)
    df['target_multiclass'] = pd.cut(
        df['returns'],
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2]  # 0=down, 1=neutral, 2=up
    ).astype(int)
    
    # Regression target: next day's return
    df['target_regression'] = df.groupby('ticker')['returns'].shift(-1)
    
    print(f"\n📊 Target Distribution (Binary):")
    print(df['target_binary'].value_counts())
    print(f"  Up: {(df['target_binary']==1).sum()/len(df)*100:.1f}%")
    print(f"  Down: {(df['target_binary']==0).sum()/len(df)*100:.1f}%")
    
    print(f"\n📊 Target Distribution (Multiclass):")
    print(df['target_multiclass'].value_counts().sort_index())
    print(f"  Down: {(df['target_multiclass']==0).sum()/len(df)*100:.1f}%")
    print(f"  Neutral: {(df['target_multiclass']==1).sum()/len(df)*100:.1f}%")
    print(f"  Up: {(df['target_multiclass']==2).sum()/len(df)*100:.1f}%")
    
    print(f"\n📊 Target Statistics (Regression):")
    print(df['target_regression'].describe())
    
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
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        if model_type == 'binary':
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        else:
            models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, objective='multi:softmax')
    
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


def train_regression_models(X_train, y_train, X_test, y_test):
    """Train and evaluate regression models"""
    print("\n" + "="*60)
    print("REGRESSION MODELS")
    print("="*60)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
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
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n📊 {name} Results:")
        print(f"  Train MAE:  {train_mae:.6f}")
        print(f"  Test MAE:   {test_mae:.6f}")
        print(f"  Train RMSE: {train_rmse:.6f}")
        print(f"  Test RMSE:  {test_rmse:.6f}")
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test R²:    {test_r2:.4f}")
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test
        }
    
    return results


def save_best_models(results_binary, results_multiclass, results_regression, 
                     scaler, feature_cols):
    """Save the best models"""
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best classification model (by F1 score)
    best_binary = max(results_binary.items(), key=lambda x: x[1]['test_f1'])
    best_binary_name, best_binary_result = best_binary
    
    binary_path = models_dir / f'classifier_binary_{timestamp}.pkl'
    joblib.dump(best_binary_result['model'], binary_path)
    print(f"✓ Saved best binary classifier: {best_binary_name}")
    print(f"  Path: {binary_path}")
    print(f"  Test F1: {best_binary_result['test_f1']:.4f}")
    
    # Save best multiclass model
    best_multi = max(results_multiclass.items(), key=lambda x: x[1]['test_f1'])
    best_multi_name, best_multi_result = best_multi
    
    multi_path = models_dir / f'classifier_multiclass_{timestamp}.pkl'
    joblib.dump(best_multi_result['model'], multi_path)
    print(f"\n✓ Saved best multiclass classifier: {best_multi_name}")
    print(f"  Path: {multi_path}")
    print(f"  Test F1: {best_multi_result['test_f1']:.4f}")
    
    # Save best regression model (by MAE)
    best_reg = min(results_regression.items(), key=lambda x: x[1]['test_mae'])
    best_reg_name, best_reg_result = best_reg
    
    reg_path = models_dir / f'regressor_{timestamp}.pkl'
    joblib.dump(best_reg_result['model'], reg_path)
    print(f"\n✓ Saved best regressor: {best_reg_name}")
    print(f"  Path: {reg_path}")
    print(f"  Test MAE: {best_reg_result['test_mae']:.6f}")
    
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
            'test_f1': float(best_binary_result['test_f1']),
            'path': str(binary_path)
        },
        'multiclass_classifier': {
            'name': best_multi_name,
            'test_accuracy': float(best_multi_result['test_acc']),
            'test_f1': float(best_multi_result['test_f1']),
            'path': str(multi_path)
        },
        'regressor': {
            'name': best_reg_name,
            'test_mae': float(best_reg_result['test_mae']),
            'test_r2': float(best_reg_result['test_r2']),
            'path': str(reg_path)
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
    df = create_target_variables(df, threshold=0.001)
    
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
    
    # Train multiclass classification models
    y_train_multi = train_df['target_multiclass']
    y_test_multi = test_df['target_multiclass']
    results_multiclass = train_classification_models(
        X_train_scaled, y_train_multi, 
        X_test_scaled, y_test_multi, 
        model_type='multiclass'
    )
    
    # Train regression models
    train_df_reg = train_df.dropna(subset=['target_regression'])
    test_df_reg = test_df.dropna(subset=['target_regression'])
    
    X_train_reg = scaler.transform(train_df_reg[feature_cols])
    X_test_reg = scaler.transform(test_df_reg[feature_cols])
    y_train_reg = train_df_reg['target_regression']
    y_test_reg = test_df_reg['target_regression']
    
    results_regression = train_regression_models(
        X_train_reg, y_train_reg, 
        X_test_reg, y_test_reg
    )
    
    # Save best models
    metadata = save_best_models(
        results_binary, results_multiclass, results_regression,
        scaler, feature_cols
    )
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return metadata


if __name__ == "__main__":
    main()
