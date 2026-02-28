"""
Example: Targeted analysis on Apple (AAPL)
Fetches market data and news specific to Apple

USAGE:
    python docs/examples/apple_analysis_example.py

NOTE: This will download data. Run only when needed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import DataIngestion
import pandas as pd

def analyze_apple():
    """Complete analysis of Apple data"""
    
    print("=" * 60)
    print("APPLE (AAPL) ANALYSIS")
    print("=" * 60)
    
    # Initialize data ingestion
    ingestion = DataIngestion(data_dir="data")
    
    # Define Apple keywords
    apple_keywords = ['Apple', 'AAPL', 'iPhone', 'iPad', 'Mac', 'Tim Cook']
    
    print("\n1. FETCHING APPLE MARKET DATA...")
    print("-" * 60)
    
    # Fetch Apple market data
    market_data = ingestion.fetch_market_data(
        tickers=['AAPL'],
        period='1y'  # 1 year of data
    )
    
    if not market_data.empty:
        print(f"SUCCESS: {len(market_data)} days of data retrieved")
        print(f"\nPeriod: {market_data['Date'].min()} to {market_data['Date'].max()}")
        print(f"\nLatest values:")
        print(market_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail())
        
        # Statistics
        print(f"\nSTATISTICS:")
        print(f"  - Average price: ${market_data['Close'].mean():.2f}")
        print(f"  - Min price: ${market_data['Close'].min():.2f}")
        print(f"  - Max price: ${market_data['Close'].max():.2f}")
        print(f"  - Volatility (std): ${market_data['Close'].std():.2f}")
    
    print("\n\n2. FETCHING APPLE NEWS...")
    print("-" * 60)
    
    # Method 1: News from yfinance (FREE - recommended)
    print("\n2.1. News from yfinance:")
    news_yf = ingestion.fetch_news_from_yfinance('AAPL')
    
    if not news_yf.empty:
        print(f"SUCCESS: {len(news_yf)} articles retrieved")
        print("\nLatest articles:")
        for idx, row in news_yf.head(5).iterrows():
            print(f"\n  - {row['title']}")
            print(f"    Source: {row['publisher']} | Date: {row['published']}")
            print(f"    Link: {row['link']}")
    else:
        print("WARNING: No news found")
    
    # Method 2: Filter Kaggle dataset (if available)
    print("\n\n2.2. Filtering Kaggle dataset by Apple keywords:")
    try:
        kaggle_news = ingestion.load_kaggle_financial_news()
        apple_news = ingestion.filter_news_by_company(
            kaggle_news, 
            company_keywords=apple_keywords
        )
        
        if not apple_news.empty:
            print(f"SUCCESS: {len(apple_news)} Apple-related articles found")
            print(f"\nExamples:")
            print(apple_news.head())
        else:
            print("WARNING: No Apple articles found in dataset")
    except FileNotFoundError:
        print("WARNING: Kaggle dataset not found")
        print("  Download from: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
    
    # Method 3: Use fetch_company_data (all-in-one)
    print("\n\n3. COMPLETE METHOD (all-in-one):")
    print("-" * 60)
    
    apple_data = ingestion.fetch_company_data(
        ticker='AAPL',
        company_keywords=apple_keywords,
        market_period='6mo'
    )
    
    print(f"\nSUCCESS: Data retrieved:")
    print(f"  - Market data: {len(apple_data['market_data'])} records")
    print(f"  - Yfinance news: {len(apple_data['news_yfinance'])} articles")
    print(f"  - Filtered news (Kaggle): {len(apple_data['news_filtered'])} articles")
    
    # Save summary
    print("\n\n4. SAVING DATA...")
    print("-" * 60)
    
    summary = {
        'ticker': 'AAPL',
        'analysis_date': pd.Timestamp.now(),
        'market_records': len(market_data),
        'news_count': len(news_yf),
        'date_range': f"{market_data['Date'].min()} to {market_data['Date'].max()}" if not market_data.empty else 'N/A'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = Path("data/processed/apple_analysis_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"SUCCESS: Summary saved: {summary_path}")
    print(f"SUCCESS: Market data: data/processed/market_data_raw.csv")
    print(f"SUCCESS: News: data/processed/news_AAPL_yfinance.csv")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return apple_data


def example_with_newsapi():
    """
    Example with NewsAPI (requires free API key)
    Visit https://newsapi.org/ to get a free key
    """
    print("\n\nEXAMPLE WITH NEWSAPI (optional):")
    print("-" * 60)
    print("To use NewsAPI:")
    print("1. Create an account at https://newsapi.org/")
    print("2. Get your free API key")
    print("3. Uncomment the code below and add your key\n")
    
    # Uncomment and add your API key:
    # API_KEY = "your_api_key_here"
    # ingestion = DataIngestion()
    # news = ingestion.fetch_news_from_newsapi(
    #     query='Apple OR AAPL',
    #     api_key=API_KEY,
    #     from_date='2024-01-01'
    # )
    # print(f"Articles retrieved: {len(news)}")


if __name__ == "__main__":
    print("\n⚠️  This example will download Apple data from the internet.")
    print("Make sure you have:")
    print("  1. Kaggle dataset in data/raw/all-data.csv")
    print("  2. Internet connection for yfinance")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() == 'y':
        # Run Apple analysis
        apple_data = analyze_apple()
        
        # Display NewsAPI example
        example_with_newsapi()
    else:
        print("Example cancelled.")
