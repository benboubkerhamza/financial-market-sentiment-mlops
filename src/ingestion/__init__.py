"""
Data Ingestion Package for Financial Market Sentiment MLOps

This package handles data collection from multiple sources:
- Financial news sentiment data
- Market data from yfinance API
- Company-specific news from various sources
"""

from .orchestrator import DataIngestion
from .market_data import MarketDataFetcher
from .news_sources import NewsSourcesFetcher

__all__ = ['DataIngestion', 'MarketDataFetcher', 'NewsSourcesFetcher']
