"""
News Sources Fetching Module

Handles fetching news from multiple sources:
- yfinance news
- NewsAPI
- Kaggle dataset filtering
"""

import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
from typing import List
from .base import BaseIngestion


class NewsSourcesFetcher(BaseIngestion):
    """Class to handle news fetching from various sources"""

    def fetch_news_from_yfinance(self, ticker: str) -> pd.DataFrame:
        """
        Fetch news for a specific ticker using yfinance

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            DataFrame containing news articles with title, publisher, link, and published date
        """
        self.logger.info(f"Fetching news for {ticker} from yfinance")

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
                self.logger.info(f"Retrieved {len(df)} news articles for {ticker}")

                # Save to processed directory
                output_path = self.processed_dir / f"news_{ticker}_yfinance.csv"
                df.to_csv(output_path, index=False)

                return df
            else:
                self.logger.warning(f"No news found for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {str(e)}")
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
        self.logger.info(f"Fetching news from NewsAPI for query: {query}")

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

            response = requests.get(url, params=params, timeout=10)
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
                self.logger.info(f"Retrieved {len(df)} articles from NewsAPI")

                # Save to processed directory
                output_path = self.processed_dir / f"news_newsapi_{query.replace(' ', '_')}.csv"
                df.to_csv(output_path, index=False)

                return df
            else:
                self.logger.warning(f"No articles found for query: {query}")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()

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

        self.logger.info(f"Loading financial news data from {dataset_path}")
        df = pd.read_csv(dataset_path, encoding='utf-8', encoding_errors='ignore')

        # Save to processed directory
        output_path = self.processed_dir / "financial_news_raw.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(df)} news records to {output_path}")

        return df

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
        self.logger.info(f"Filtering news by keywords: {company_keywords}")

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
        self.logger.info(f"Filtered {len(news_df)} news down to {len(filtered_df)} relevant articles")

        return filtered_df

    def load_financial_phrase_bank(
        self,
        agreement_level: str = "50",
        dataset_path: str = None
    ) -> pd.DataFrame:
        """
        Load FinancialPhraseBank dataset with sentiment labels

        This is a high-quality sentiment analysis dataset with 4,840 financial sentences
        annotated by experts from Aalto University.

        Args:
            agreement_level: Level of annotator agreement
                           "50" (4,840 sentences), "66", "75", or "all" (216 sentences)
            dataset_path: Path to FinancialPhraseBank directory
                         If None, looks in data/raw/FinancialPhraseBank/

        Returns:
            DataFrame with columns: ['sentence', 'sentiment']
        """
        self.logger.info(f"Loading FinancialPhraseBank (agreement_level={agreement_level})")

        # Determine file path
        if dataset_path is None:
            base_path = self.raw_dir / "FinancialPhraseBank"
        else:
            base_path = Path(dataset_path)

        # Map agreement level to filename
        file_mapping = {
            "50": "Sentences_50Agree.txt",
            "66": "Sentences_66Agree.txt",
            "75": "Sentences_75Agree.txt",
            "all": "Sentences_AllAgree.txt"
        }

        if agreement_level not in file_mapping:
            raise ValueError(f"Invalid agreement_level: {agreement_level}. Must be one of {list(file_mapping.keys())}")

        file_path = base_path / file_mapping[agreement_level]

        if not file_path.exists():
            raise FileNotFoundError(
                f"FinancialPhraseBank file not found: {file_path}\n"
                f"Download from: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news"
            )

        # Parse the file
        sentences = []
        sentiments = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: "sentence text@sentiment"
                if '@' in line:
                    text, sentiment = line.rsplit('@', 1)
                    sentences.append(text.strip())
                    sentiments.append(sentiment.strip())

        df = pd.DataFrame({
            'sentence': sentences,
            'sentiment': sentiments
        })

        self.logger.info(f"Loaded {len(df)} sentences from FinancialPhraseBank")
        self.logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

        # Save to processed directory
        output_path = self.processed_dir / f"financial_phrase_bank_{agreement_level}.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved to {output_path}")

        return df
