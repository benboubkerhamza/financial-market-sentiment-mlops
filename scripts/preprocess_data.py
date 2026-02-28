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
    pipeline = DataPreprocessingPipeline(data_dir='data/processed')
    
    # Run preprocessing
    try:
        processed_data = pipeline.run_pipeline(save_output=True)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        
        # News data summary
        news_df = processed_data['news']
        print(f"\nNews Data:")
        print(f"  - Total records: {len(news_df)}")
        print(f"  - Columns: {list(news_df.columns)}")
        print(f"  - Average text length: {news_df['processed_text_length'].mean():.1f} words")
        
        # Sentiment data summary
        sentiment_df = processed_data['sentiment']
        print(f"\nSentiment Data:")
        print(f"  - Total records: {len(sentiment_df)}")
        print(f"  - Columns: {list(sentiment_df.columns)}")
        print(f"  - Sentiment distribution:")
        if 'sentiment_encoded' in sentiment_df.columns:
            sentiment_counts = sentiment_df['sentiment_encoded'].value_counts()
            for sentiment_val, count in sentiment_counts.items():
                sentiment_label = {1: 'positive', -1: 'negative', 0: 'neutral'}.get(sentiment_val, 'unknown')
                print(f"    * {sentiment_label}: {count} ({count/len(sentiment_df)*100:.1f}%)")
        
        # Market data summary
        market_df = processed_data['market']
        print(f"\nMarket Data:")
        print(f"  - Total records: {len(market_df)}")
        print(f"  - Number of features: {len(market_df.columns)}")
        print(f"  - Tickers: {market_df['Ticker'].nunique() if 'Ticker' in market_df.columns else 'N/A'}")
        print(f"  - Date range: {market_df['Date'].min()} to {market_df['Date'].max()}")
        print(f"  - Missing values: {market_df.isnull().sum().sum()}")
        
        # Technical indicators summary
        technical_indicators = [col for col in market_df.columns if any(
            indicator in col for indicator in ['ma_', 'ema_', 'rsi', 'macd', 'bb_', 'volatility']
        )]
        print(f"\nTechnical Indicators ({len(technical_indicators)}):")
        for indicator in technical_indicators[:10]:  # Show first 10
            print(f"  - {indicator}")
        if len(technical_indicators) > 10:
            print(f"  ... and {len(technical_indicators) - 10} more")
        
        print("\n" + "=" * 80)
        print("Preprocessing completed successfully!")
        print("Processed files saved in: data/processed/")
        print("  - news_processed.csv")
        print("  - sentiment_processed.csv")
        print("  - market_processed.csv")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
