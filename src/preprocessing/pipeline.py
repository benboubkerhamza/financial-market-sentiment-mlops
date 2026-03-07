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

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize preprocessing pipeline.

        Args:
            data_dir: Base data directory (default: 'data')
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.text_preprocessor = TextPreprocessor()
        self.market_preprocessor = MarketDataPreprocessor()
        self.feature_engineer = FeatureEngineering()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw data files.

        Returns:
            Tuple of (market_df, sentiment_training_df, news_df)
            - market_df: Market data with prices and volume
            - sentiment_training_df: FinancialPhraseBank for training sentiment model
            - news_df: Real news articles with timestamps
        """
        # Load market data from raw directory
        market_df = pd.read_csv(self.raw_dir / 'market_data_raw.csv')
        # Load sentiment training data (FinancialPhraseBank)
        sentiment_training_df = pd.read_csv(self.raw_dir / 'all-data.csv')
        # Load real news articles with timestamps
        news_df = pd.read_csv(self.raw_dir / 'financial_news_raw.csv')

        return market_df, sentiment_training_df, news_df

    def preprocess_sentiment_training_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess sentiment training data (FinancialPhraseBank).

        Args:
            sentiment_df: Sentiment phrases DataFrame with 'text' column

        Returns:
            Processed sentiment DataFrame
        """
        # Preprocess sentiment phrases
        sentiment_processed = self.text_preprocessor.preprocess_dataframe(
            sentiment_df,
            text_column='text',
            output_column='processed_text'
        )

        # Encode sentiments
        sentiment_processed = self.feature_engineer.encode_sentiment(sentiment_processed)

        return sentiment_processed

    def preprocess_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess news articles data.

        Args:
            news_df: News DataFrame with 'title' and 'summary' columns

        Returns:
            Processed news DataFrame
        """
        news_processed = news_df.copy()
        # Combine title and summary for full text
        if 'title' in news_df.columns and 'summary' in news_df.columns:
            news_processed['full_text'] = news_df['title'].fillna('') + ' ' + news_df['summary'].fillna('')
        elif 'title' in news_df.columns:
            news_processed['full_text'] = news_df['title'].fillna('')
        elif 'summary' in news_df.columns:
            news_processed['full_text'] = news_df['summary'].fillna('')
        else:
            raise ValueError("News DataFrame must have 'title' or 'summary' column")
        # Preprocess combined text
        news_processed = self.text_preprocessor.preprocess_dataframe(
            news_processed,
            text_column='full_text',
            output_column='processed_text'
        )
        return news_processed

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
        print("\n" + "="*80)
        print("STEP 1: Loading raw data...")
        print("="*80)
        market_df, sentiment_training_df, news_df = self.load_data()
        print(f"✓ Loaded {len(market_df)} market records")
        print(f"✓ Loaded {len(sentiment_training_df)} sentiment training phrases (FinancialPhraseBank)")
        print(f"✓ Loaded {len(news_df)} news articles")

        # Preprocess sentiment training data (FinancialPhraseBank)
        print("\n" + "="*80)
        print("STEP 2: Preprocessing sentiment training data (FinancialPhraseBank)...")
        print("="*80)
        sentiment_processed = self.preprocess_sentiment_training_data(sentiment_training_df)
        print(f"✓ Processed {len(sentiment_processed)} sentiment phrases")

        # Preprocess news articles
        print("\n" + "="*80)
        print("STEP 3: Preprocessing news articles...")
        print("="*80)
        news_processed = self.preprocess_news_data(news_df)
        print(f"✓ Processed {len(news_processed)} news articles")

        # Preprocess market data
        print("\n" + "="*80)
        print("STEP 4: Preprocessing market data and calculating technical indicators...")
        print("="*80)
        market_processed = self.preprocess_market_data(market_df)
        print(f"✓ Processed market data: {len(market_processed)} records with {len(market_processed.columns)} features")

        # Handle missing values
        print("\n" + "="*80)
        print("STEP 5: Handling missing values...")
        print("="*80)
        market_processed = self.feature_engineer.handle_missing_values(market_processed)
        print(f"✓ Missing values handled")

        # Save processed data to processed directory
        if save_output:
            print("\n" + "="*80)
            print("STEP 6: Saving processed data...")
            print("="*80)
            sentiment_processed.to_csv(self.processed_dir / 'sentiment_training_processed.csv', index=False)
            print(f"  ✓ sentiment_training_processed.csv (FinancialPhraseBank)")
            news_processed.to_csv(self.processed_dir / 'news_processed.csv', index=False)
            print(f"  ✓ news_processed.csv (News articles)")
            market_processed.to_csv(self.processed_dir / 'market_processed.csv', index=False)
            print(f"  ✓ market_processed.csv (Market data)")
            print(f"\nAll files saved to {self.processed_dir}/")

        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE COMPLETED!")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Train sentiment model using sentiment_training_processed.csv")
        print("  2. Apply sentiment model to news_processed.csv")
        print("  3. Merge sentiment predictions with market_processed.csv")
        print("  4. Train final ML model for market prediction")
        print("="*80 + "\n")

        return {
            'sentiment_training': sentiment_processed,
            'news': news_processed,
            'market': market_processed
        }

        print("\nPreprocessing pipeline completed successfully!")

        return {
            'sentiment': sentiment_processed,
            'market': market_processed
        }