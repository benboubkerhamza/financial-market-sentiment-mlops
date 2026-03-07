"""
Stock Price Prediction Script

Make predictions using the trained Logistic Regression model.
Predicts whether stock prices will go up or down.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_model_artifacts(timestamp='20260308_000626'):
    """Load trained model and artifacts"""
    print("="*60)
    print("LOADING MODEL ARTIFACTS")
    print("="*60)
    
    models_dir = project_root / 'models'
    
    # Load metadata
    metadata_path = models_dir / f'metadata_{timestamp}.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load binary classifier (best model)
    model_path = models_dir / f'classifier_binary_{timestamp}.pkl'
    model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = models_dir / f'scaler_{timestamp}.pkl'
    scaler = joblib.load(scaler_path)
    
    # Load feature list
    features_path = models_dir / f'features_{timestamp}.json'
    with open(features_path, 'r') as f:
        feature_data = json.load(f)
        feature_cols = feature_data['features']
    
    print(f"✓ Loaded model: {metadata['binary_classifier']['name']}")
    print(f"  Accuracy: {metadata['binary_classifier']['test_accuracy']:.2%}")
    print(f"  F1 Score: {metadata['binary_classifier']['test_f1']:.4f}")
    print(f"✓ Loaded {len(feature_cols)} features")
    
    return model, scaler, feature_cols, metadata


def load_latest_market_data():
    """Load the latest market data"""
    print("\n" + "="*60)
    print("LOADING MARKET DATA")
    print("="*60)
    
    data_path = project_root / 'data' / 'processed' / 'market_processed.csv'
    df = pd.read_csv(data_path)
    
    # Normalize column names
    df.rename(columns={'Date': 'date', 'Ticker': 'ticker'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Tickers: {df['ticker'].nunique()} unique - {', '.join(sorted(df['ticker'].unique()))}")
    
    return df


def get_latest_data_per_ticker(df, n_days=1):
    """Get the most recent N days of data for each ticker"""
    latest_data = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date', ascending=False)
        latest_data.append(ticker_data.head(n_days))
    
    result = pd.concat(latest_data, ignore_index=True)
    return result.sort_values(['ticker', 'date'])


def make_predictions(model, scaler, feature_cols, data):
    """Make predictions on data"""
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Check for missing features
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(f"⚠️  Warning: Missing features: {missing_features}")
        return None
    
    # Prepare features
    X = data[feature_cols].copy()
    
    # Check for missing values
    if X.isna().any().any():
        print("⚠️  Warning: Data contains NaN values. Filling with 0...")
        X = X.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Add predictions to data
    data = data.copy()
    data['prediction'] = predictions
    data['prediction_label'] = data['prediction'].map({0: 'DOWN ⬇️', 1: 'UP ⬆️'})
    data['confidence_down'] = probabilities[:, 0]
    data['confidence_up'] = probabilities[:, 1]
    data['confidence'] = np.max(probabilities, axis=1)
    
    print(f"✓ Made predictions for {len(data)} records")
    
    return data


def display_predictions(predictions_df):
    """Display predictions in a user-friendly format"""
    print("\n" + "="*60)
    print("📊 STOCK PRICE PREDICTIONS")
    print("="*60)
    
    for _, row in predictions_df.iterrows():
        ticker = row['ticker']
        date = row['date'].strftime('%Y-%m-%d')
        prediction = row['prediction_label']
        confidence = row['confidence']
        confidence_up = row['confidence_up']
        confidence_down = row['confidence_down']
        
        # Color coding
        if prediction == 'UP ⬆️':
            symbol = '📈'
        else:
            symbol = '📉'
        
        print(f"\n{symbol} {ticker} - {date}")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Prob(UP): {confidence_up:.1%} | Prob(DOWN): {confidence_down:.1%}")
        
        # Add current technical indicators context
        if 'rsi' in predictions_df.columns:
            rsi = row['rsi']
            rsi_signal = 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
            print(f"   RSI: {rsi:.1f} ({rsi_signal})")
        
        if 'macd' in predictions_df.columns and 'macd_signal' in predictions_df.columns:
            macd = row['macd']
            macd_signal = row['macd_signal']
            macd_trend = 'Bullish' if macd > macd_signal else 'Bearish'
            print(f"   MACD: {macd_trend}")


def save_predictions(predictions_df, output_path=None):
    """Save predictions to CSV"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / 'data' / 'predictions' / f'predictions_{timestamp}.csv'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select relevant columns
    output_cols = [
        'ticker', 'date', 'prediction_label', 'confidence',
        'confidence_up', 'confidence_down'
    ]
    
    # Add technical indicators if available
    for col in ['rsi', 'macd', 'macd_signal', 'returns', 'volatility']:
        if col in predictions_df.columns:
            output_cols.append(col)
    
    predictions_df[output_cols].to_csv(output_path, index=False)
    
    print(f"\n✓ Saved predictions to: {output_path}")
    
    return output_path


def predict_for_tomorrow():
    """Predict stock movements for tomorrow based on today's data"""
    print("\n")
    print("="*60)
    print("🔮 STOCK PRICE PREDICTION FOR TOMORROW")
    print("="*60)
    
    # Load model
    model, scaler, feature_cols, metadata = load_model_artifacts()
    
    # Load market data
    df = load_latest_market_data()
    
    # Get latest data (today's data to predict tomorrow)
    latest_data = get_latest_data_per_ticker(df, n_days=1)
    
    print(f"\n📅 Prediction Date: Tomorrow ({(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')})")
    print(f"📅 Based on data from: {latest_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Make predictions
    predictions = make_predictions(model, scaler, feature_cols, latest_data)
    
    if predictions is not None:
        # Display predictions
        display_predictions(predictions)
        
        # Summary statistics
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        up_count = (predictions['prediction'] == 1).sum()
        down_count = (predictions['prediction'] == 0).sum()
        avg_confidence = predictions['confidence'].mean()
        
        print(f"\n  Bullish predictions (UP): {up_count}/{len(predictions)} ({up_count/len(predictions)*100:.1f}%)")
        print(f"  Bearish predictions (DOWN): {down_count}/{len(predictions)} ({down_count/len(predictions)*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.1%}")
        
        # Save predictions
        output_path = save_predictions(predictions)
        
        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETED")
        print("="*60)
        
        return predictions
    else:
        print("\n❌ Could not make predictions due to missing features")
        return None


def predict_specific_ticker(ticker, model_timestamp='20260308_000626'):
    """Predict for a specific ticker"""
    print("\n")
    print("="*60)
    print(f"🔮 PREDICTION FOR {ticker}")
    print("="*60)
    
    # Load model
    model, scaler, feature_cols, metadata = load_model_artifacts(model_timestamp)
    
    # Load market data
    df = load_latest_market_data()
    
    # Filter for specific ticker
    ticker_data = df[df['ticker'] == ticker].sort_values('date', ascending=False).head(1)
    
    if len(ticker_data) == 0:
        print(f"❌ No data found for ticker {ticker}")
        return None
    
    # Make prediction
    predictions = make_predictions(model, scaler, feature_cols, ticker_data)
    
    if predictions is not None:
        display_predictions(predictions)
        return predictions
    else:
        print("\n❌ Could not make prediction")
        return None


def main():
    """Main prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, help='Specific ticker to predict (e.g., AAPL)')
    parser.add_argument('--model', type=str, default='20260308_000626', 
                       help='Model timestamp to use')
    
    args = parser.parse_args()
    
    if args.ticker:
        # Predict for specific ticker
        predict_specific_ticker(args.ticker.upper(), args.model)
    else:
        # Predict for all tickers
        predict_for_tomorrow()


if __name__ == "__main__":
    main()
