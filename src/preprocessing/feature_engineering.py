"""
Feature Engineering Module

Handles feature creation including sentiment encoding, lag features,
rolling window features, and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineering:
    """
    Creates features from preprocessed data for machine learning.
    """

    def __init__(self):
        """Initialize feature engineering."""
        pass

    def encode_sentiment(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """
        Encode sentiment labels to numeric values.

        Args:
            df: DataFrame with sentiment column
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with encoded sentiment
        """
        df = df.copy()

        # Map sentiment to numeric values
        sentiment_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }

        df[f'{sentiment_column}_encoded'] = df[sentiment_column].map(sentiment_map)

        # Handle any unmapped values
        df[f'{sentiment_column}_encoded'] = df[f'{sentiment_column}_encoded'].fillna(0)

        return df

    def create_lag_features(self, df: pd.DataFrame, columns: List[str],
                           lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, columns: List[str],
                               windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max).

        Args:
            df: DataFrame with time series data
            columns: Columns to create rolling features for
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with potential missing values
            strategy: Strategy to handle missing values ('forward_fill', 'backward_fill', 'drop', 'mean')

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        if strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        return df
