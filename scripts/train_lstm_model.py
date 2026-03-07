"""
LSTM Model Training with Sliding Window

Train LSTM models for stock price prediction using sliding window approach.
This captures temporal patterns better than simple feature-based models.
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for stock price prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Pass through fully connected layer
        output = self.fc(last_output)
        
        return output


def load_and_prepare_data():
    """Load market data"""
    print("="*60)
    print("DATA LOADING")
    print("="*60)
    
    data_path = project_root / 'data' / 'processed' / 'market_processed.csv'
    df = pd.read_csv(data_path)
    
    # Normalize column names
    df.rename(columns={'Date': 'date', 'Ticker': 'ticker'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()} unique")
    
    return df


def select_features(df):
    """Select features for LSTM"""
    feature_cols = [
        'returns',
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
    print(f"Total features: {len(available_features)}")
    
    return available_features


def create_sequences(data, ticker_col, feature_cols, window_size=30, forecast_horizon=1):
    """
    Create sliding window sequences for LSTM.
    
    Args:
        data: DataFrame with features
        ticker_col: Column name for ticker grouping
        feature_cols: List of feature columns
        window_size: Number of time steps to look back (default: 30)
        forecast_horizon: Number of steps to predict ahead (default: 1)
    
    Returns:
        X: Array of shape (n_samples, window_size, n_features)
        y: Array of shape (n_samples,) - target returns
        metadata: List of dicts with ticker and date info
    """
    print("\n" + "="*60)
    print("CREATING SLIDING WINDOW SEQUENCES")
    print("="*60)
    print(f"  Window size: {window_size} days")
    print(f"  Forecast horizon: {forecast_horizon} day(s)")
    
    X_list = []
    y_list = []
    metadata_list = []
    
    # Process each ticker separately
    for ticker in data[ticker_col].unique():
        ticker_data = data[data[ticker_col] == ticker].sort_values('date').reset_index(drop=True)
        
        # Get feature values
        feature_values = ticker_data[feature_cols].values
        
        # Create target (next day's return)
        target_returns = ticker_data['returns'].shift(-forecast_horizon).values
        
        # Create sequences
        for i in range(window_size, len(ticker_data) - forecast_horizon + 1):
            # Sequence of past window_size days
            sequence = feature_values[i-window_size:i]
            
            # Target is the return forecast_horizon days ahead
            target = target_returns[i-1]  # Index i-1 because shift already moved it
            
            if not np.isnan(target) and not np.isnan(sequence).any():
                X_list.append(sequence)
                y_list.append(target)
                metadata_list.append({
                    'ticker': ticker,
                    'date': ticker_data.loc[i, 'date'],
                    'sequence_start': ticker_data.loc[i-window_size, 'date'],
                    'sequence_end': ticker_data.loc[i-1, 'date']
                })
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✓ Created {len(X)} sequences")
    print(f"  Shape: {X.shape} (samples, time_steps, features)")
    print(f"  Targets shape: {y.shape}")
    print(f"  Tickers: {len(data[ticker_col].unique())}")
    
    return X, y, metadata_list


def temporal_train_test_split(X, y, metadata, test_size=0.2):
    """Split sequences temporally"""
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT (TEMPORAL)")
    print("="*60)
    
    # Sort by date to ensure temporal split
    dates = [m['date'] for m in metadata]
    sorted_indices = np.argsort(dates)
    
    split_idx = int(len(sorted_indices) * (1 - test_size))
    
    train_indices = sorted_indices[:split_idx]
    test_indices = sorted_indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    metadata_train = [metadata[i] for i in train_indices]
    metadata_test = [metadata[i] for i in test_indices]
    
    print(f"Train set: {len(X_train)} sequences")
    print(f"  Date range: {metadata_train[0]['date']} to {metadata_train[-1]['date']}")
    print(f"Test set: {len(X_test)} sequences")
    print(f"  Date range: {metadata_test[0]['date']} to {metadata_test[-1]['date']}")
    print(f"Split ratio: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    
    return X_train, X_test, y_train, y_test, metadata_train, metadata_test


def scale_sequences(X_train, X_test):
    """Scale features in sequences"""
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)
    
    # Reshape to 2D for scaling
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    print("✓ Scaled sequences using StandardScaler")
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_lstm(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu'):
    """Train LSTM model"""
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\n✓ Training completed")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses


def evaluate_regression(model, test_loader, device='cpu'):
    """Evaluate LSTM regression model"""
    print("\n" + "="*60)
    print("LSTM REGRESSION EVALUATION")
    print("="*60)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    print(f"\n📊 LSTM Regression Results:")
    print(f"  Test MAE:  {mae:.6f}")
    print(f"  Test RMSE: {rmse:.6f}")
    print(f"  Test R²:   {r2:.4f}")
    
    # Classification metrics (predict direction)
    pred_direction = (predictions > 0).astype(int)
    actual_direction = (actuals > 0).astype(int)
    
    accuracy = accuracy_score(actual_direction, pred_direction)
    precision = precision_score(actual_direction, pred_direction)
    recall = recall_score(actual_direction, pred_direction)
    f1 = f1_score(actual_direction, pred_direction)
    
    print(f"\n📊 Direction Prediction (Binary Classification):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'direction_accuracy': accuracy,
        'direction_f1': f1,
        'predictions': predictions,
        'actuals': actuals
    }


def save_model(model, scaler, feature_cols, window_size, results):
    """Save LSTM model and artifacts"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save PyTorch model
    model_path = models_dir / f'lstm_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': 1
        }
    }, model_path)
    print(f"✓ Saved LSTM model: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / f'lstm_scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'LSTM',
        'window_size': window_size,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'test_mae': float(results['mae']),
        'test_rmse': float(results['rmse']),
        'test_r2': float(results['r2']),
        'direction_accuracy': float(results['direction_accuracy']),
        'direction_f1': float(results['direction_f1']),
        'model_path': str(model_path),
        'scaler_path': str(scaler_path)
    }
    
    metadata_path = models_dir / f'lstm_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    return metadata


def main():
    """Main training pipeline"""
    print("\n")
    print("="*60)
    print("LSTM STOCK PREDICTION - SLIDING WINDOW APPROACH")
    print("="*60)
    
    # Configuration
    WINDOW_SIZE = 30  # 30 days lookback
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Select features
    feature_cols = select_features(df)
    
    # Remove rows with missing values
    df_clean = df.dropna(subset=feature_cols + ['returns'])
    print(f"\n✓ After removing missing values: {len(df_clean)} records")
    
    # Create sequences
    X, y, metadata = create_sequences(
        df_clean, 
        ticker_col='ticker',
        feature_cols=feature_cols,
        window_size=WINDOW_SIZE,
        forecast_horizon=1
    )
    
    # Split temporally
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = temporal_train_test_split(
        X, y, metadata, test_size=0.2
    )
    
    # Scale sequences
    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n✓ Created DataLoaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    input_size = X_train_scaled.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.2,
        output_size=1
    )
    
    print(f"\n✓ Created LSTM model")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    
    # Train model
    model, train_losses, val_losses = train_lstm(
        model, train_loader, test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Evaluate
    results = evaluate_regression(model, test_loader, device=device)
    
    # Save model
    metadata = save_model(model, scaler, feature_cols, WINDOW_SIZE, results)
    
    print("\n" + "="*60)
    print("✓ LSTM TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  Direction Accuracy: {results['direction_accuracy']:.4f}")
    
    return metadata


if __name__ == "__main__":
    main()
