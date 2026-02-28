"""
Preprocessing Pipeline Module

Main orchestration pipeline that combines text preprocessing,
market data preprocessing, and feature engineering.
"""

import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

from .text_preprocessor import TextPreprocessor
from .market_preprocessor import MarketDataPreprocessor
from .feature_engineering import FeatureEngineering


class DataPreprocessingPipeline:
    """
    Main preprocessing pipeline that orchestrates all preprocessing steps.
    """

    def __init__(self, data_dir: str = 'data/processed'):
        """
        Initialize preprocessing pipeline.

        Args:
            data_dir: Directory containing raw data
        """
        self.data_dir = Path(data_dir)
        self.text_preprocessor = TextPreprocessor()
        self.market_preprocessor = MarketDataPreprocessor()
        self.feature_engineer = FeatureEngineering()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all raw data files.

        Returns:
            Tuple of (news_df, market_df, sentiment_df)
        """
        news_df = pd.read_csv(self.data_dir / 'financial_news_raw.csv')
        market_df = pd.read_csv(self.data_dir / 'market_data_raw.csv')
        sentiment_df = pd.read_csv(self.data_dir / 'financial_phrase_bank_50.csv')

        return news_df, market_df, sentiment_df

    def preprocess_text_data(self, news_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess all text data.

        Args:
            news_df: News articles DataFrame
            sentiment_df: Sentiment phrases DataFrame

        Returns:
            Tuple of (news_processed, sentiment_processed)
        """
        # Preprocess news articles
        news_processed = self.text_preprocessor.preprocess_dataframe(
            news_df,
            text_column='text',
            output_column='processed_text'
        )

        # Preprocess sentiment phrases
        sentiment_processed = self.text_preprocessor.preprocess_dataframe(
            sentiment_df,
            text_column='sentence',
            output_column='processed_text'
        )

        # Encode sentiments
        news_processed = self.feature_engineer.encode_sentiment(news_processed)
        sentiment_processed = self.feature_engineer.encode_sentiment(sentiment_processed)

        return news_processed, sentiment_processed

    def preprocess_market_data(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data with technical indicators.

        Args:
            market_df: Raw market data

        Returns:
            Processed market data with indicators
        """
        # Convert Date column to datetime
        market_df['Date'] = pd.to_datetime(market_df['Date'])

        # Process market data
        market_processed = self.market_preprocessor.preprocess_market_data(market_df)

        # Create additional features
        feature_columns = ['returns', 'volatility', 'rsi', 'macd']
        market_processed = self.feature_engineer.create_lag_features(
            market_processed,
            columns=feature_columns,
            lags=[1, 2, 3]
        )

        return market_processed

    def run_pipeline(self, save_output: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.

        Args:
            save_output: Whether to save processed data to disk

        Returns:
            Dictionary containing all processed DataFrames
        """
        print("Starting preprocessing pipeline...")

        # Load data
        print("Loading raw data...")
        news_df, market_df, sentiment_df = self.load_data()
        print(f"Loaded {len(news_df)} news articles, {len(market_df)} market records, "
              f"{len(sentiment_df)} sentiment phrases")

        # Preprocess text data
        print("\nPreprocessing text data...")
        news_processed, sentiment_processed = self.preprocess_text_data(news_df, sentiment_df)
        print(f"Processed text data: {len(news_processed)} news, {len(sentiment_processed)} sentiment phrases")

        # Preprocess market data
        print("\nPreprocessing market data and calculating technical indicators...")
        market_processed = self.preprocess_market_data(market_df)
        print(f"Processed market data: {len(market_processed)} records with {len(market_processed.columns)} features")

        # Handle missing values
        print("\nHandling missing values...")
        market_processed = self.feature_engineer.handle_missing_values(market_processed)

        # Save processed data
        if save_output:
            print("\nSaving processed data...")
            news_processed.to_csv(self.data_dir / 'news_processed.csv', index=False)
            sentiment_processed.to_csv(self.data_dir / 'sentiment_processed.csv', index=False)
            market_processed.to_csv(self.data_dir / 'market_processed.csv', index=False)
            print("Processed data saved to data/processed/")

        print("\nPreprocessing pipeline completed successfully!")

        return {
            'news': news_processed,
            'sentiment': sentiment_processed,
            'market': market_processed
        }
