"""
Data Collection Script

Execute full data ingestion pipeline to collect:
- Kaggle Financial News dataset
- Market data for major tech stocks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import DataIngestion


def main():
    """Execute full data collection"""
    print("=" * 60)
    print("FINANCIAL MARKET SENTIMENT - DATA COLLECTION")
    print("=" * 60)
    
    # Initialize ingestion
    ingestion = DataIngestion(data_dir="data")
    
    # Define tickers to collect
    tickers = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'META',   # Meta (Facebook)
        'TSLA',   # Tesla
        'NVDA',   # NVIDIA
        'JPM',    # JPMorgan
        'V',      # Visa
        'WMT',    # Walmart
    ]
    
    print(f"\nCollecting data for {len(tickers)} companies...")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    # Execute data ingestion
    sentiment_df, news_df, market_df = ingestion.ingest_all_data(
        tickers=tickers,
        market_period='2y'  # 2 years of historical data
    )
    
    # Display FinancialPhraseBank info (already loaded by ingest_all_data)
    print("\n" + "=" * 60)
    print("FINANCIALPHRASEBANK DATASET (SENTIMENT TRAINING DATA)")
    print("=" * 60)
    
    if not sentiment_df.empty:
        print(f"\n[FINANCIALPHRASEBANK]")
        print(f"  Total sentences: {len(sentiment_df):,}")
        print(f"  Columns: {list(sentiment_df.columns)}")
        if 'sentiment' in sentiment_df.columns:
            print(f"  Sentiment distribution:")
            for sentiment, count in sentiment_df['sentiment'].value_counts().items():
                print(f"    {sentiment}: {count:,} ({count/len(sentiment_df)*100:.1f}%)")
        print(f"  Note: This dataset is for training the sentiment model")
    else:
        print(f"\n[FINANCIALPHRASEBANK] - Not loaded (check dataset path)")
    
    # Display results
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    if not news_df.empty:
        print(f"\n[NEWS DATA - Real Articles with Timestamps]")
        print(f"  Total articles: {len(news_df):,}")
        print(f"  Columns: {list(news_df.columns)}")
        if 'ticker' in news_df.columns:
            print(f"  Tickers: {news_df['ticker'].unique().tolist()}")
        if 'published' in news_df.columns:
            print(f"  Date range: {news_df['published'].min()} to {news_df['published'].max()}")
        print(f"  Saved to: data/raw/financial_news_raw.csv")
        print(f"  Note: These news will be used to predict sentiment + merge with market data")
        print(f"\n  Sample:")
        print(news_df[['ticker', 'title', 'published']].head(3).to_string() if 'ticker' in news_df.columns else news_df.head(3).to_string())
    else:
        print("\n[NEWS DATA] - No data collected")
    
    if not market_df.empty:
        print(f"\n[MARKET DATA]")
        print(f"  Total records: {len(market_df):,}")
        # Get tickers from DataFrame (could be 'ticker' or 'Ticker')
        ticker_col = 'Ticker' if 'Ticker' in market_df.columns else 'ticker'
        print(f"  Tickers: {market_df[ticker_col].unique().tolist()}")
        print(f"  Columns: {list(market_df.columns)}")
        print(f"  Date range: {market_df.index.min()} to {market_df.index.max()}")
        print(f"  Saved to: data/raw/market_data_raw.csv")

        # Statistics per ticker
        print(f"\n  Records per ticker:")
        for ticker in tickers:
            count = len(market_df[market_df[ticker_col] == ticker])
            print(f"    {ticker}: {count:,} records")
    else:
        print("\n[MARKET DATA] - No data collected")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED!")
    print("=" * 60)
    
    # Summary statistics
    total_datasets = 0
    if not sentiment_df.empty:
        total_datasets += 1
    if not news_df.empty:
        total_datasets += 1
    if not market_df.empty:
        total_datasets += 1
    
    print(f"\nTotal datasets collected: {total_datasets}/3")
    print(f"  - FinancialPhraseBank (for sentiment training): {'✓' if not sentiment_df.empty else '✗'}")
    print(f"  - News Articles (with timestamps): {'✓' if not news_df.empty else '✗'}")
    print(f"  - Market Data: {'✓' if not market_df.empty else '✗'}")
    
    # Next steps
    print("\n" + "=" * 60)
    print("DATA ARCHITECTURE")
    print("=" * 60)
    print("\n1. FinancialPhraseBank → Train sentiment model (FinBERT/VADER)")
    print("2. News Articles → Apply sentiment model → Get sentiment scores")
    print("3. Sentiment scores + Market Data → Merge by (date, ticker)")
    print("4. Final dataset → Train ML model to predict market movements")
    print("\nNext Steps:")
    print("  1. Check data quality in data/raw/ and data/processed/")
    print("  2. Run exploratory data analysis (EDA)")
    print("  3. Implement sentiment model training (Phase 2.2)")
    print("  4. Apply sentiment to news articles")
    print("  5. Merge sentiment + market data for final ML training")


if __name__ == "__main__":
    main()
