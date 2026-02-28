"""
Data Ingestion Module for Financial Market Sentiment MLOps

This module handles data collection from multiple sources:
- Financial news sentiment data from Kaggle
- Market data from yfinance API
- Company-specific news from various sources
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging
import requests
from typing import List, Dict, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Class to handle data ingestion from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataIngestion class
        
        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_kaggle_financial_news(self, dataset_path: str = None) -> pd.DataFrame:
        """
        Load financial news sentiment data from Kaggle dataset
        
        Args:
            dataset_path: Path to the Kaggle dataset CSV file
                         If None, looks for common filenames in data/raw/
        
        Returns:
            DataFrame containing financial news sentiment data
        """
        if dataset_path is None:
            # Try common Kaggle financial news dataset filenames
            possible_files = [
                self.raw_dir / "all-data.csv",
                self.raw_dir / "financial_news.csv",
                self.raw_dir / "financial_sentiment.csv",
                self.raw_dir / "FinancialPhraseBank-v1.0.csv"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    dataset_path = file_path
                    break
            
            if dataset_path is None:
                raise FileNotFoundError(
                    f"No Kaggle dataset found in {self.raw_dir}. "
                    "Please download the 'Financial News Sentiment' dataset from Kaggle "
                    "and place it in the data/raw/ directory."
                )
        
        logger.info(f"Loading financial news data from {dataset_path}")
        df = pd.read_csv(dataset_path, encoding='utf-8', encoding_errors='ignore')
        
        # Save to processed directory
        output_path = self.processed_dir / "financial_news_raw.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} news records to {output_path}")
        
        return df
    
    def fetch_market_data(
        self, 
        tickers: list,
        start_date: str = None,
        end_date: str = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch market data using yfinance API
        
        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if start_date/end_date not provided
                   (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        
        Returns:
            DataFrame containing market data with OHLCV information
        """
        logger.info(f"Fetching market data for tickers: {tickers}")
        
        market_data = []
        
        for ticker in tickers:
            try:
                logger.info(f"Downloading data for {ticker}")
                stock = yf.Ticker(ticker)
                
                if start_date and end_date:
                    df = stock.history(start=start_date, end=end_date)
                else:
                    df = stock.history(period=period)
                
                if not df.empty:
                    df['Ticker'] = ticker
                    df.reset_index(inplace=True)
                    market_data.append(df)
                    logger.info(f"Retrieved {len(df)} records for {ticker}")
                else:
                    logger.warning(f"No data retrieved for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        if market_data:
            combined_df = pd.concat(market_data, ignore_index=True)
            
            # Save to processed directory
            output_path = self.processed_dir / "market_data_raw.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(combined_df)} market records to {output_path}")
            
            return combined_df
        else:
            logger.warning("No market data retrieved for any ticker")
            return pd.DataFrame()
    
    def fetch_news_from_yfinance(self, ticker: str) -> pd.DataFrame:
        """
        Fetch news for a specific ticker using yfinance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        
        Returns:
            DataFrame containing news articles with title, publisher, link, and published date
        """
        logger.info(f"Fetching news for {ticker} from yfinance")
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                news_data = []
                for article in news:
                    news_data.append({
                        'ticker': ticker,
                        'title': article.get('title', ''),
                        'publisher': article.get('publisher', ''),
                        'link': article.get('link', ''),
                        'published': pd.to_datetime(article.get('providerPublishTime'), unit='s') if article.get('providerPublishTime') else None,
                        'type': article.get('type', ''),
                        'thumbnail': article.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if article.get('thumbnail') else ''
                    })
                
                df = pd.DataFrame(news_data)
                logger.info(f"Retrieved {len(df)} news articles for {ticker}")
                
                # Save to processed directory
                output_path = self.processed_dir / f"news_{ticker}_yfinance.csv"
                df.to_csv(output_path, index=False)
                
                return df
            else:
                logger.warning(f"No news found for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_news_from_newsapi(
        self, 
        query: str, 
        api_key: str,
        from_date: str = None,
        to_date: str = None,
        language: str = 'en'
    ) -> pd.DataFrame:
        """
        Fetch news using NewsAPI (requires API key from newsapi.org)
        
        Args:
            query: Search query (e.g., 'Apple OR AAPL')
            api_key: NewsAPI key (get free key at newsapi.org)
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
            language: Language code (default: 'en')
        
        Returns:
            DataFrame containing news articles
        """
        logger.info(f"Fetching news from NewsAPI for query: {query}")
        
        try:
            url = 'https://newsapi.org/v2/everything'
            
            params = {
                'q': query,
                'apiKey': api_key,
                'language': language,
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            if articles:
                news_data = []
                for article in articles:
                    news_data.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'author': article.get('author', ''),
                        'url': article.get('url', ''),
                        'published': pd.to_datetime(article.get('publishedAt')),
                        'urlToImage': article.get('urlToImage', '')
                    })
                
                df = pd.DataFrame(news_data)
                logger.info(f"Retrieved {len(df)} articles from NewsAPI")
                
                # Save to processed directory
                output_path = self.processed_dir / f"news_newsapi_{query.replace(' ', '_')}.csv"
                df.to_csv(output_path, index=False)
                
                return df
            else:
                logger.warning(f"No articles found for query: {query}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()
    
    def filter_news_by_company(
        self, 
        news_df: pd.DataFrame, 
        company_keywords: List[str],
        text_column: str = None
    ) -> pd.DataFrame:
        """
        Filter news dataset by company-related keywords
        
        Args:
            news_df: DataFrame containing news data
            company_keywords: List of keywords to search for (e.g., ['Apple', 'AAPL', 'iPhone'])
            text_column: Name of the column containing text to search
                        If None, searches all string columns
        
        Returns:
            Filtered DataFrame containing only relevant news
        """
        logger.info(f"Filtering news by keywords: {company_keywords}")
        
        if news_df.empty:
            return news_df
        
        # Create case-insensitive pattern
        pattern = '|'.join(company_keywords)
        
        if text_column and text_column in news_df.columns:
            # Filter specific column
            mask = news_df[text_column].astype(str).str.contains(pattern, case=False, na=False)
        else:
            # Search all string columns
            mask = pd.Series([False] * len(news_df))
            for col in news_df.select_dtypes(include=['object']).columns:
                mask |= news_df[col].astype(str).str.contains(pattern, case=False, na=False)
        
        filtered_df = news_df[mask].copy()
        logger.info(f"Filtered {len(news_df)} news down to {len(filtered_df)} relevant articles")
        
        return filtered_df
    
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
        logger.info(f"Fetching complete data for {ticker}")
        
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
            logger.warning(f"Could not filter Kaggle news: {str(e)}")
            result['news_filtered'] = pd.DataFrame()
        
        # 4. Fetch from NewsAPI if requested
        if use_newsapi and newsapi_key:
            query = ' OR '.join(company_keywords)
            news_api = self.fetch_news_from_newsapi(query, newsapi_key)
            result['news_api'] = news_api
        else:
            result['news_api'] = pd.DataFrame()
        
        # Summary
        logger.info(f"\n=== Data Summary for {ticker} ===")
        logger.info(f"Market data: {len(market_df)} records")
        logger.info(f"News (yfinance): {len(news_yf)} articles")
        logger.info(f"News (filtered Kaggle): {len(result['news_filtered'])} articles")
        logger.info(f"News (NewsAPI): {len(result['news_api'])} articles")
        
        return result
    
    def fetch_sp500_tickers(self) -> list:
        """
        Get list of S&P 500 ticker symbols
        
        Returns:
            List of ticker symbols
        """
        try:
            # Get S&P 500 tickers from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            # Return some common tickers as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    
    def ingest_all_data(
        self,
        news_dataset_path: str = None,
        tickers: list = None,
        market_period: str = "1y"
    ) -> tuple:
        """
        Ingest all data sources (news and market data)
        
        Args:
            news_dataset_path: Path to Kaggle financial news dataset
            tickers: List of tickers to fetch market data for
            market_period: Period for market data
        
        Returns:
            Tuple of (news_df, market_df)
        """
        logger.info("Starting full data ingestion pipeline")
        
        # Load news data
        try:
            news_df = self.load_kaggle_financial_news(news_dataset_path)
        except Exception as e:
            logger.error(f"Failed to load news data: {str(e)}")
            news_df = pd.DataFrame()
        s
        # Fetch market data
        if tickers is None:
            logger.info("No tickers provided, using default popular stocks")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
        
        try:
            market_df = self.fetch_market_data(tickers, period=market_period)
        except Exception as e:
            logger.error(f"Failed to fetch market data: {str(e)}")
            market_df = pd.DataFrame()
        
        logger.info("Data ingestion pipeline completed")
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
