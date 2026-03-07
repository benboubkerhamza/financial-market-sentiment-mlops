"""
Script to execute the complete data preprocessing pipeline.

This script:
1. Loads raw data from data/processed/
2. Preprocesses text (news articles and sentiment phrases)
3. Calculates technical indicators for market data
4. Engineers features for machine learning
5. Saves processed data back to data/processed/
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import DataPreprocessingPipeline


def main():
    """Execute the preprocessing pipeline."""
    print("=" * 80)
    print("Financial Market Sentiment - Data Preprocessing Pipeline")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DataPreprocessingPipeline(data_dir='data')
    
    # Run preprocessing
    try:
        processed_data = pipeline.run_pipeline(save_output=True)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        
        # Sentiment training data summary
        sentiment_training_df = processed_data['sentiment_training']
        print(f"\n1. Sentiment Training Data (FinancialPhraseBank):")
        print(f"   - Total records: {len(sentiment_training_df)}")
        print(f"   - Columns: {list(sentiment_training_df.columns)}")
        print(f"   - Usage: Train sentiment model (FinBERT/VADER)")
        if 'sentiment_encoded' in sentiment_training_df.columns:
            sentiment_counts = sentiment_training_df['sentiment_encoded'].value_counts()
            print(f"   - Sentiment distribution:")
            for sentiment_val, count in sentiment_counts.items():
                sentiment_label = {1: 'positive', -1: 'negative', 0: 'neutral'}.get(sentiment_val, 'unknown')
                print(f"     * {sentiment_label}: {count} ({count/len(sentiment_training_df)*100:.1f}%)")
        # News articles summary
        news_df = processed_data['news']
        print(f"\n2. News Articles (with timestamps):")
        print(f"   - Total records: {len(news_df)}")
        print(f"   - Columns: {list(news_df.columns)}")
        print(f"   - Usage: Apply sentiment model → Get sentiment scores → Merge with market")
        if 'ticker' in news_df.columns:
            print(f"   - Tickers: {news_df['ticker'].nunique()} unique ({', '.join(news_df['ticker'].unique()[:5].tolist())}...)")
        if 'published' in news_df.columns:
            print(f"   - Date range: {news_df['published'].min()} to {news_df['published'].max()}")
        
        # Market data summary
        market_df = processed_data['market']
        print(f"\n3. Market Data:")
        print(f"   - Total records: {len(market_df)}")
        print(f"   - Number of features: {len(market_df.columns)}")
        print(f"   - Usage: Merge with sentiment predictions → Train final ML model")
        if 'Ticker' in market_df.columns:
            print(f"   - Tickers: {market_df['Ticker'].nunique()} unique")
        if 'Date' in market_df.columns:
            print(f"   - Date range: {market_df['Date'].min()} to {market_df['Date'].max()}")
        print(f"   - Missing values: {market_df.isnull().sum().sum()}")
        
        # Technical indicators summary
        technical_indicators = [col for col in market_df.columns if any(
            indicator in col for indicator in ['ma_', 'ema_', 'rsi', 'macd', 'bb_', 'volatility']
        )]
        print(f"\n4. Technical Indicators ({len(technical_indicators)}):")
        for indicator in technical_indicators[:10]:  # Show first 10
            print(f"   - {indicator}")
        if len(technical_indicators) > 10:
            print(f"   ... and {len(technical_indicators) - 10} more")
        
        print("\n" + "=" * 80)
        print("PROCESSED FILES SAVED")
        print("=" * 80)
        print("  ✓ sentiment_training_processed.csv (for training sentiment model)")
        print("  ✓ news_processed.csv (for applying sentiment predictions)")
        print("  ✓ market_processed.csv (for final ML model)")
        print("\n  Location: data/processed/")
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("  1. Implement sentiment model (FinBERT/VADER)")
        print("  2. Train sentiment model on sentiment_training_processed.csv")
        print("  3. Apply sentiment model to news_processed.csv")
        print("  4. Merge sentiment predictions with market_processed.csv by (date, ticker)")
        print("  5. Create final dataset for ML model training")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
