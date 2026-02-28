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
    news_df, market_df = ingestion.ingest_all_data(
        tickers=tickers,
        market_period='2y'  # 2 years of historical data
    )
    
    # Load FinancialPhraseBank dataset
    print("\n" + "=" * 60)
    print("LOADING FINANCIALPHRASEBANK DATASET")
    print("=" * 60)
    
    try:
        phrase_bank_df = ingestion.load_financial_phrase_bank(agreement_level="50")
        print(f"\n[FINANCIALPHRASEBANK]")
        print(f"  Total sentences: {len(phrase_bank_df):,}")
        print(f"  Columns: {list(phrase_bank_df.columns)}")
        print(f"  Sentiment distribution:")
        for sentiment, count in phrase_bank_df['sentiment'].value_counts().items():
            print(f"    {sentiment}: {count:,} ({count/len(phrase_bank_df)*100:.1f}%)")
        print(f"  Saved to: data/processed/financial_phrase_bank_50.csv")
    except Exception as e:
        print(f"\n[FINANCIALPHRASEBANK] - Error: {str(e)}")
        phrase_bank_df = None
    
    # Display results
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    if not news_df.empty:
        print(f"\n[NEWS DATA]")
        print(f"  Total articles: {len(news_df):,}")
        print(f"  Columns: {list(news_df.columns)}")
        print(f"  Saved to: data/processed/financial_news_raw.csv")
        print(f"\n  Sample:")
        print(news_df.head(3).to_string())
    else:
        print("\n[NEWS DATA] - No data collected (check Kaggle dataset path)")
    
    if not market_df.empty:
        print(f"\n[MARKET DATA]")
        print(f"  Total records: {len(market_df):,}")
        # Get tickers from DataFrame (could be 'ticker' or 'Ticker')
        ticker_col = 'Ticker' if 'Ticker' in market_df.columns else 'ticker'
        print(f"  Tickers: {market_df[ticker_col].unique().tolist()}")
        print(f"  Columns: {list(market_df.columns)}")
        print(f"  Date range: {market_df.index.min()} to {market_df.index.max()}")
        print(f"  Saved to: data/processed/market_data_raw.csv")

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
    if not news_df.empty:
        total_datasets += 1
    if not market_df.empty:
        total_datasets += 1
    if phrase_bank_df is not None:
        total_datasets += 1
    
    print(f"\nTotal datasets collected: {total_datasets}/3")
    print(f"  - Kaggle News: {'✓' if not news_df.empty else '✗'}")
    print(f"  - Market Data: {'✓' if not market_df.empty else '✗'}")
    print(f"  - FinancialPhraseBank: {'✓' if phrase_bank_df is not None else '✗'}")
    
    # Next steps
    print("\nNext Steps:")
    print("  1. Check data quality in data/processed/")
    print("  2. Run exploratory data analysis (EDA)")
    print("  3. Start preprocessing pipeline")
    print("  4. Implement sentiment analysis using FinancialPhraseBank as training data")


if __name__ == "__main__":
    main()
