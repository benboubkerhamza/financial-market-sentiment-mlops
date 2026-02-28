# financial-market-sentiment-mlops
End-to-end MLOps pipeline for financial news sentiment analysis and market movement prediction. Includes model training, experiment tracking, API deployment, containerization, and orchestration.

## 2. Data Sources

### 2.1 Financial News
- **Dataset**: Kaggle Financial News Sentiment
- **Description**: Dataset containing financial news with associated sentiments
- **Installation**:
  1. Download the dataset from Kaggle: [Financial Phrase Bank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
  2. Place the CSV file in the `data/raw/` folder
- **Supported formats**: `all-data.csv`, `financial_news.csv`, `financial_sentiment.csv`

### 2.2 Market Data
- **API**: yfinance (Python)
- **Description**: API to retrieve historical financial market data
- **Available data**: OHLCV (Open, High, Low, Close, Volume)
- **Supported tickers**: Individual stocks, S&P 500 indices

### 2.3 Usage

```python
from src.ingestion import DataIngestion

# Initialize data ingestion
ingestion = DataIngestion(data_dir="data")

# Load financial news
news_df = ingestion.load_kaggle_financial_news()

# Fetch market data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
market_df = ingestion.fetch_market_data(
    tickers=tickers,
    period='1y'  # 1 year of historical data
)

# Or ingest all data at once
news_df, market_df = ingestion.ingest_all_data(
    tickers=tickers,
    market_period='1y'
)
```

### 2.4 Data Structure

**News data**:
- Financial news text
- Associated sentiment (positive, negative, neutral)
- Metadata (date, source, etc.)

**Market data**:
- Date
- Ticker (stock symbol)
- Open, High, Low, Close (OHLC)
- Volume
- Dividends and splits
