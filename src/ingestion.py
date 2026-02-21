"""
Data ingestion module for financial news sentiment MLOps pipeline.

Fetches financial news articles from public APIs / RSS feeds and
stores the raw data as CSV files inside the `data/` directory.
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

FINANCIAL_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


def fetch_alpha_vantage_news(ticker: str, limit: int = 50) -> list[dict]:
    """Fetch news sentiment data from Alpha Vantage for a given ticker."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        articles = data.get("feed", [])
        records = []
        for article in articles:
            records.append(
                {
                    "ticker": ticker,
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("time_published", ""),
                    "source": article.get("source", ""),
                    "overall_sentiment_label": article.get("overall_sentiment_label", ""),
                    "overall_sentiment_score": article.get("overall_sentiment_score", 0.0),
                }
            )
        logger.info("Fetched %d articles for %s from Alpha Vantage.", len(records), ticker)
        return records
    except requests.RequestException as exc:
        logger.error("Error fetching Alpha Vantage data for %s: %s", ticker, exc)
        return []


def fetch_newsapi_articles(query: str, days_back: int = 7, page_size: int = 50) -> list[dict]:
    """Fetch financial news from NewsAPI for a given query string."""
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set; skipping NewsAPI ingestion.")
        return []

    from datetime import timedelta
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        records = []
        for article in articles:
            records.append(
                {
                    "ticker": query,
                    "title": article.get("title", ""),
                    "summary": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "overall_sentiment_label": "",
                    "overall_sentiment_score": 0.0,
                }
            )
        logger.info("Fetched %d articles for query '%s' from NewsAPI.", len(records), query)
        return records
    except requests.RequestException as exc:
        logger.error("Error fetching NewsAPI data for '%s': %s", query, exc)
        return []


def ingest_all(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Ingest news data for all configured tickers and persist to CSV.

    Returns:
        DataFrame containing all ingested articles.
    """
    if tickers is None:
        tickers = FINANCIAL_TICKERS

    all_records: list[dict] = []
    for ticker in tickers:
        all_records.extend(fetch_alpha_vantage_news(ticker))
        all_records.extend(fetch_newsapi_articles(ticker))

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No data ingested.")
        return df

    df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(DATA_DIR, f"raw_news_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    logger.info("Saved %d records to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    ingest_all()
