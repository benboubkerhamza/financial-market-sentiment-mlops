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

    def load_financial_phrase_bank(self, agreement_level: str = "50", **kwargs) -> pd.DataFrame:
        """Load FinancialPhraseBank dataset. See NewsSourcesFetcher.load_financial_phrase_bank for details."""
        return self.news_fetcher.load_financial_phrase_bank(agreement_level, **kwargs)

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ingest all data sources (news and market data)

        Args:
            news_dataset_path: Path to Kaggle financial news dataset
            tickers: List of tickers to fetch market data for
            market_period: Period for market data

        Returns:
            Tuple of (news_df, market_df)
        """
        self.logger.info("Starting full data ingestion pipeline")

        # Load news data
        try:
            news_df = self.load_kaggle_financial_news(news_dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to load news data: {str(e)}")
            news_df = pd.DataFrame()

        # Fetch market data
        if tickers is None:
            self.logger.info("No tickers provided, using default popular stocks")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']

        try:
            market_df = self.fetch_market_data(tickers, period=market_period)
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {str(e)}")
            market_df = pd.DataFrame()

        self.logger.info("Data ingestion pipeline completed")
        return news_df, market_df


def main():
    """Main function to demonstrate data ingestion"""
    ingestion = DataIngestion()

    # Ingest all data
    news_df, market_df = ingestion.ingest_all_data(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        market_period='1y'
    )

    print(f"\n=== Data Ingestion Summary ===")
    print(f"News data shape: {news_df.shape}")
    print(f"Market data shape: {market_df.shape}")

    if not news_df.empty:
        print(f"\nNews data columns: {list(news_df.columns)}")
        print(f"News data preview:\n{news_df.head()}")

    if not market_df.empty:
        print(f"\nMarket data columns: {list(market_df.columns)}")
        print(f"Market data preview:\n{market_df.head()}")


if __name__ == "__main__":
    main()
