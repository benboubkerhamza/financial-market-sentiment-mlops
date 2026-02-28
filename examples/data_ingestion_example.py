"""
Example script demonstrating data ingestion from Kaggle and yfinance
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import DataIngestion

def main():
    print("=== Financial Market Sentiment Data Ingestion Example ===\n")
    
    # Initialize data ingestion
    ingestion = DataIngestion(data_dir="data")
    
    # Example 1: Fetch market data for specific tickers
    print("1. Fetching market data from yfinance...")
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    market_df = ingestion.fetch_market_data(
        tickers=tickers,
        period='6mo'  # 6 months of historical data
    )
    
    if not market_df.empty:
        print(f"   ✓ Retrieved {len(market_df)} market records")
        print(f"   Columns: {list(market_df.columns)}")
        print(f"\n   Sample data:")
        print(market_df.head())
    else:
        print("   ✗ No market data retrieved")
    
    # Example 2: Load Kaggle financial news data
    print("\n2. Loading financial news data from Kaggle...")
    try:
        news_df = ingestion.load_kaggle_financial_news()
        print(f"   ✓ Loaded {len(news_df)} news records")
        print(f"   Columns: {list(news_df.columns)}")
        print(f"\n   Sample data:")
        print(news_df.head())
    except FileNotFoundError as e:
        print(f"   ✗ {str(e)}")
        print("\n   To download the dataset:")
        print("   1. Go to: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
        print("   2. Download the dataset")
        print("   3. Place it in data/raw/ directory")
    
    # Example 3: Fetch data with date range
    print("\n3. Fetching market data with custom date range...")
    market_df_custom = ingestion.fetch_market_data(
        tickers=['AAPL', 'TSLA'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if not market_df_custom.empty:
        print(f"   ✓ Retrieved {len(market_df_custom)} records for 2024")
    
    print("\n=== Data Ingestion Complete ===")
    print(f"Data saved in: data/processed/")

if __name__ == "__main__":
    main()
