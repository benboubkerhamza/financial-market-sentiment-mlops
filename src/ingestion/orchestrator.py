"""
Data Ingestion Orchestrator

Main class that combines market data and news fetching
"""

import pandas as pd
from typing import List, Dict, Tuple
from .base import BaseIngestion
from .market_data import MarketDataFetcher
from .news_sources import NewsSourcesFetcher


class DataIngestion(BaseIngestion):
    """
    Main orchestrator class for data ingestion
    Combines market data and news fetching capabilities
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataIngestion orchestrator

        Args:
            data_dir: Directory to save downloaded data
        """
        super().__init__(data_dir)
        self.market_fetcher = MarketDataFetcher(data_dir)
        self.news_fetcher = NewsSourcesFetcher(data_dir)

    # Expose market data methods
    def fetch_market_data(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Fetch market data. See MarketDataFetcher.fetch_market_data for details."""
        return self.market_fetcher.fetch_market_data(tickers, **kwargs)

    def fetch_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers. See MarketDataFetcher.fetch_sp500_tickers for details."""
        return self.market_fetcher.fetch_sp500_tickers()

    def get_company_info(self, ticker: str) -> dict:
        """Get company info. See MarketDataFetcher.get_company_info for details."""
        return self.market_fetcher.get_company_info(ticker)

    # Expose news methods
    def fetch_news_from_yfinance(self, ticker: str) -> pd.DataFrame:
        """Fetch news from yfinance. See NewsSourcesFetcher.fetch_news_from_yfinance for details."""
        return self.news_fetcher.fetch_news_from_yfinance(ticker)

    def fetch_news_from_newsapi(self, query: str, api_key: str, **kwargs) -> pd.DataFrame:
        """Fetch news from NewsAPI. See NewsSourcesFetcher.fetch_news_from_newsapi for details."""
        return self.news_fetcher.fetch_news_from_newsapi(query, api_key, **kwargs)

    def load_kaggle_financial_news(self, dataset_path: str = None) -> pd.DataFrame:
        """Load Kaggle dataset. See NewsSourcesFetcher.load_kaggle_financial_news for details."""
        return self.news_fetcher.load_kaggle_financial_news(dataset_path)

    def filter_news_by_company(self, news_df: pd.DataFrame, company_keywords: List[str], **kwargs) -> pd.DataFrame:
        """Filter news by keywords. See NewsSourcesFetcher.filter_news_by_company for details."""
        return self.news_fetcher.filter_news_by_company(news_df, company_keywords, **kwargs)

    # Orchestration methods
    def fetch_company_data(
        self,
        ticker: str,
        company_keywords: List[str] = None,
        market_period: str = "1y",
        use_newsapi: bool = False,
        newsapi_key: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data for a specific company: market data + news

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            company_keywords: Keywords to filter news (e.g., ['Apple', 'AAPL', 'iPhone'])
                            If None, uses ticker symbol
            market_period: Period for market data
            use_newsapi: Whether to use NewsAPI for news
            newsapi_key: NewsAPI key (required if use_newsapi=True)

        Returns:
            Dictionary with keys: 'market_data', 'news_yfinance', 'news_filtered', 'news_api'
        """
        self.logger.info(f"Fetching complete data for {ticker}")

        result = {}

        # 1. Fetch market data
        market_df = self.fetch_market_data([ticker], period=market_period)
        result['market_data'] = market_df

        # 2. Fetch news from yfinance
        news_yf = self.fetch_news_from_yfinance(ticker)
        result['news_yfinance'] = news_yf

        # 3. Try to filter Kaggle dataset by company keywords
        if company_keywords is None:
            company_keywords = [ticker]

        try:
            kaggle_news = self.load_kaggle_financial_news()
            filtered_news = self.filter_news_by_company(kaggle_news, company_keywords)
            result['news_filtered'] = filtered_news

            # Save filtered news
            if not filtered_news.empty:
                output_path = self.processed_dir / f"news_{ticker}_filtered.csv"
                filtered_news.to_csv(output_path, index=False)
        except Exception as e:
            self.logger.warning(f"Could not filter Kaggle news: {str(e)}")
            result['news_filtered'] = pd.DataFrame()

        # 4. Fetch from NewsAPI if requested
        if use_newsapi and newsapi_key:
            query = ' OR '.join(company_keywords)
            news_api = self.fetch_news_from_newsapi(query, newsapi_key)
            result['news_api'] = news_api
        else:
            result['news_api'] = pd.DataFrame()

        # Summary
        self.logger.info(f"\n=== Data Summary for {ticker} ===")
        self.logger.info(f"Market data: {len(market_df)} records")
        self.logger.info(f"News (yfinance): {len(news_yf)} articles")
        self.logger.info(f"News (filtered Kaggle): {len(result['news_filtered'])} articles")
        self.logger.info(f"News (NewsAPI): {len(result['news_api'])} articles")

        return result

    def ingest_all_data(
        self,
        news_dataset_path: str = None,
        tickers: List[str] = None,
        market_period: str = "1y"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ingest all data sources (sentiment training data, news, and market data)

        Args:
            news_dataset_path: Path to Kaggle financial news dataset (FinancialPhraseBank)
            tickers: List of tickers to fetch market data for
            market_period: Period for market data

        Returns:
            Tuple of (sentiment_df, news_df, market_df)
            - sentiment_df: FinancialPhraseBank for training sentiment model
            - news_df: Real news articles with timestamps from yfinance
            - market_df: Market data with prices and volume
        """
        self.logger.info("Starting full data ingestion pipeline")

        # Load FinancialPhraseBank (training data for sentiment - kept separate)
        try:
            sentiment_df = self.load_kaggle_financial_news(news_dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to load news data: {str(e)}")
            sentiment_df = pd.DataFrame()

        # Fetch market data
        if tickers is None:
            self.logger.info("No tickers provided, using default popular stocks")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']

        market_df = pd.DataFrame()
        news_df = pd.DataFrame()
        try:
            # Fetch market data for all tickers
            self.logger.info(f"Fetching market data for tickers: {tickers}")
            market_df = self.fetch_market_data(tickers, period=market_period)
            # Fetch news from yfinance for ALL tickers
            self.logger.info(f"Fetching news for all {len(tickers)} tickers")
            news_list = []
            for ticker in tickers:
                try:
                    ticker_news = self.fetch_news_from_yfinance(ticker)
                    if not ticker_news.empty:
                        news_list.append(ticker_news)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch news for {ticker}: {str(e)}")
            # Combine all news into one DataFrame
            if news_list:
                news_df = pd.concat(news_list, ignore_index=True)
                # Save combined news to raw directory
                output_path = self.raw_dir / "financial_news_raw.csv"
                news_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved {len(news_df)} news records to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to fetch market/news data: {str(e)}")

        self.logger.info("Data ingestion pipeline completed")
        self.logger.info(f"\n=== Final Data Summary ===")
        self.logger.info(f"Sentiment training data: {len(sentiment_df)} phrases")
        self.logger.info(f"News articles: {len(news_df)} articles")
        self.logger.info(f"Market data: {len(market_df)} records")
        return sentiment_df, news_df, market_df
