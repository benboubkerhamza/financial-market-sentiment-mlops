"""
Feature Combination Script

Merges sentiment data from news with market data to create final training dataset.
Creates lag features and handles missing values.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_data():
    """Load processed datasets"""
    data_dir = project_root / 'data' / 'processed'
    
    print("Loading datasets...")
    
    # Load market data
    market_path = data_dir / 'market_processed.csv'
    market_df = pd.read_csv(market_path)
    # Normalize column names
    market_df.rename(columns={'Date': 'date', 'Ticker': 'ticker'}, inplace=True)
    # Normalize date to remove time and timezone
    market_df['date'] = pd.to_datetime(market_df['date'], utc=True).dt.normalize()
    print(f"  Loaded {len(market_df)} market records")
    print(f"  Date range: {market_df['date'].min()} to {market_df['date'].max()}")
    print(f"  Tickers: {market_df['ticker'].nunique()} unique")
    
    # Load sentiment data
    sentiment_path = data_dir / 'news_with_sentiment.csv'
    sentiment_df = pd.read_csv(sentiment_path)
    # Convert to datetime and normalize to date only (remove time and timezone)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['published'], utc=True).dt.normalize()
    print(f"  Loaded {len(sentiment_df)} news articles with sentiment")
    print(f"  Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print(f"  Tickers: {sentiment_df['ticker'].nunique()} unique")
    
    return market_df, sentiment_df


def aggregate_daily_sentiment(sentiment_df):
    """
    Aggregate multiple news articles per day per ticker into daily sentiment metrics.
    
    For each ticker-date, compute:
    - Average sentiment scores
    - Dominant sentiment label (most frequent)
    - News count
    """
    print("\nAggregating daily sentiment...")
    
    # Group by ticker and date
    agg_dict = {
        'vader_positive': 'mean',
        'vader_negative': 'mean',
        'vader_neutral': 'mean',
        'vader_compound': 'mean',
        'vader_sentiment': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral',
        'finbert_positive': 'mean',
        'finbert_negative': 'mean',
        'finbert_neutral': 'mean',
        'finbert_sentiment': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral'
    }
    
    daily_sentiment = sentiment_df.groupby(['ticker', 'date']).agg(agg_dict).reset_index()
    
    # Add news count per day
    news_count = sentiment_df.groupby(['ticker', 'date']).size().reset_index(name='news_count')
    daily_sentiment = daily_sentiment.merge(news_count, on=['ticker', 'date'], how='left')
    
    print(f"  Aggregated to {len(daily_sentiment)} unique ticker-date combinations")
    print(f"  Average news per day: {daily_sentiment['news_count'].mean():.2f}")
    
    return daily_sentiment


def create_sentiment_lag_features(df, sentiment_cols, lags=[1, 2, 3]):
    """
    Create lag features for sentiment (previous days' sentiment).
    
    Args:
        df: DataFrame with ticker, date, and sentiment columns
        sentiment_cols: List of sentiment column names to create lags for
        lags: List of lag periods (default: [1, 2, 3])
    """
    print(f"\nCreating lag features for {len(sentiment_cols)} sentiment metrics...")
    print(f"  Lag periods: {lags}")
    
    df = df.sort_values(['ticker', 'date'])
    
    for col in sentiment_cols:
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby('ticker')[col].shift(lag)
            
    print(f"  Created {len(sentiment_cols) * len(lags)} lag features")
    
    return df


def merge_sentiment_with_market(market_df, sentiment_df):
    """
    Merge daily sentiment data with market data.
    
    Strategy: Left join market data with sentiment (market data is primary)
    """
    print("\nMerging sentiment with market data...")
    print(f"  Market records: {len(market_df)}")
    print(f"  Sentiment records: {len(sentiment_df)}")
    
    # Merge on ticker and date
    merged_df = market_df.merge(
        sentiment_df,
        on=['ticker', 'date'],
        how='left',
        suffixes=('', '_sentiment')
    )
    
    print(f"  Merged records: {len(merged_df)}")
    
    # Check merge quality
    sentiment_match_rate = (merged_df['news_count'].notna().sum() / len(merged_df)) * 100
    print(f"  Records with sentiment: {merged_df['news_count'].notna().sum()} ({sentiment_match_rate:.1f}%)")
    
    return merged_df


def handle_missing_sentiment(df):
    """
    Handle missing sentiment values.
    
    Strategy:
    - Fill missing news_count with 0
    - Fill missing sentiment scores with neutral values
    - Fill missing sentiment labels with 'neutral'
    """
    print("\nHandling missing sentiment values...")
    
    # Count missing values
    sentiment_cols = [col for col in df.columns if 'vader' in col or 'finbert' in col or col == 'news_count']
    missing_counts = df[sentiment_cols].isna().sum()
    
    if missing_counts.sum() > 0:
        print(f"  Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Fill news_count with 0
        df['news_count'] = df['news_count'].fillna(0)
        
        # Fill sentiment scores with neutral values
        neutral_fills = {
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 1.0,
            'vader_compound': 0.0,
            'finbert_positive': 0.0,
            'finbert_negative': 0.0,
            'finbert_neutral': 1.0
        }
        
        for col, value in neutral_fills.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        
        # Fill sentiment labels with 'neutral'
        for col in df.columns:
            if 'sentiment' in col and df[col].dtype == 'object':
                df[col] = df[col].fillna('neutral')
        
        print(f"  ✓ Missing values filled with neutral defaults")
    else:
        print(f"  ✓ No missing sentiment values")
    
    return df


def add_sentiment_change_features(df):
    """
    Add features tracking sentiment changes over time.
    """
    print("\nAdding sentiment change features...")
    
    df = df.sort_values(['ticker', 'date'])
    
    # Sentiment change from previous day
    df['vader_compound_change'] = df.groupby('ticker')['vader_compound'].diff()
    df['finbert_positive_change'] = df.groupby('ticker')['finbert_positive'].diff()
    
    # Sentiment momentum (3-day moving average)
    df['vader_compound_ma3'] = df.groupby('ticker')['vader_compound'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['finbert_positive_ma3'] = df.groupby('ticker')['finbert_positive'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # News volume change
    df['news_count_change'] = df.groupby('ticker')['news_count'].diff()
    
    print(f"  ✓ Added 5 sentiment change features")
    
    return df


def validate_final_dataset(df):
    """
    Validate the final combined dataset.
    """
    print("\n" + "="*60)
    print("FINAL DATASET VALIDATION")
    print("="*60)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Records: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    
    print(f"\n📅 Date Range:")
    print(f"   Start: {df['date'].min()}")
    print(f"   End: {df['date'].max()}")
    print(f"   Days: {(df['date'].max() - df['date'].min()).days + 1}")
    
    print(f"\n🏢 Tickers: {df['ticker'].nunique()}")
    print(f"   {', '.join(sorted(df['ticker'].unique()))}")
    
    print(f"\n📰 News Coverage:")
    print(f"   Days with news: {(df['news_count'] > 0).sum()} ({(df['news_count'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"   Total news articles: {df['news_count'].sum():.0f}")
    print(f"   Avg news per day: {df['news_count'].mean():.2f}")
    
    print(f"\n💭 Sentiment Distribution (VADER):")
    print(df['vader_sentiment'].value_counts())
    
    print(f"\n💭 Sentiment Distribution (FinBERT):")
    print(df['finbert_sentiment'].value_counts())
    
    print(f"\n🔢 Missing Values:")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   ✓ No missing values!")
    
    print(f"\n📈 Target Variable (returns):")
    if 'returns' in df.columns:
        print(df['returns'].describe())
    else:
        print("  Warning: 'returns' column not found")
    
    print(f"\n✅ Feature Categories:")
    market_features = [col for col in df.columns if any(x in col for x in ['price', 'volume', 'ma_', 'ema_', 'rsi', 'macd', 'bb_', 'volatility', 'return'])]
    sentiment_features = [col for col in df.columns if 'vader' in col or 'finbert' in col or 'news' in col]
    temporal_features = [col for col in df.columns if any(x in col for x in ['day_', 'month', 'quarter', 'year'])]
    
    print(f"   Market features: {len(market_features)}")
    print(f"   Sentiment features: {len(sentiment_features)}")
    print(f"   Temporal features: {len(temporal_features)}")
    print(f"   Total: {len(market_features) + len(sentiment_features) + len(temporal_features)}")
    
    return df


def main():
    """Main execution"""
    print("="*60)
    print("FEATURE COMBINATION PIPELINE")
    print("="*60)
    
    # Load data
    market_df, sentiment_df = load_data()
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    # Create sentiment lag features
    sentiment_numeric_cols = [
        'vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound',
        'finbert_positive', 'finbert_negative', 'finbert_neutral'
    ]
    daily_sentiment = create_sentiment_lag_features(daily_sentiment, sentiment_numeric_cols, lags=[1, 2, 3])
    
    # Merge sentiment with market data
    combined_df = merge_sentiment_with_market(market_df, daily_sentiment)
    
    # Handle missing sentiment values
    combined_df = handle_missing_sentiment(combined_df)
    
    # Add sentiment change features
    combined_df = add_sentiment_change_features(combined_df)
    
    # Validate final dataset
    combined_df = validate_final_dataset(combined_df)
    
    # Save final dataset
    output_path = project_root / 'data' / 'processed' / 'final_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved final dataset: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Save feature list for reference
    feature_list_path = project_root / 'data' / 'processed' / 'feature_list.txt'
    with open(feature_list_path, 'w') as f:
        f.write("FINAL DATASET FEATURES\n")
        f.write("="*60 + "\n\n")
        
        f.write("MARKET FEATURES:\n")
        market_features = [col for col in combined_df.columns if any(x in col for x in ['price', 'volume', 'ma_', 'ema_', 'rsi', 'macd', 'bb_', 'volatility', 'return'])]
        for feat in sorted(market_features):
            f.write(f"  - {feat}\n")
        
        f.write("\nSENTIMENT FEATURES:\n")
        sentiment_features = [col for col in combined_df.columns if 'vader' in col or 'finbert' in col or 'news' in col]
        for feat in sorted(sentiment_features):
            f.write(f"  - {feat}\n")
        
        f.write("\nTEMPORAL FEATURES:\n")
        temporal_features = [col for col in combined_df.columns if any(x in col for x in ['day_', 'month', 'quarter', 'year'])]
        for feat in sorted(temporal_features):
            f.write(f"  - {feat}\n")
        
        f.write("\nIDENTIFIER COLUMNS:\n")
        id_cols = ['ticker', 'date']
        for feat in id_cols:
            f.write(f"  - {feat}\n")
    
    print(f"✓ Saved feature list: {feature_list_path}")
    
    print("\n" + "="*60)
    print("✓ FEATURE COMBINATION PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
