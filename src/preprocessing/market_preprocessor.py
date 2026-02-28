"""
Market Data Preprocessing Module

Handles market data preprocessing and technical indicator calculations
including moving averages, RSI, MACD, Bollinger Bands, and volatility.
"""

import pandas as pd
import numpy as np
from typing import List


class MarketDataPreprocessor:
    """
    Preprocesses market data and calculates technical indicators.
    """

    def __init__(self):
        """Initialize market data preprocessor."""
        pass

    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate returns (percentage change).

        Args:
            df: DataFrame with market data
            price_column: Column to calculate returns from

        Returns:
            DataFrame with returns column added
        """
        df = df.copy()
        df['returns'] = df[price_column].pct_change()
        df['log_returns'] = np.log(df[price_column] / df[price_column].shift(1))
        return df

    def calculate_moving_averages(self, df: pd.DataFrame,
                                  price_column: str = 'Close',
                                  windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate simple moving averages.

        Args:
            df: DataFrame with market data
            price_column: Column to calculate MA from
            windows: List of window sizes

        Returns:
            DataFrame with MA columns added
        """
        df = df.copy()
        for window in windows:
            df[f'ma_{window}'] = df[price_column].rolling(window=window).mean()
        return df

    def calculate_exponential_moving_averages(self, df: pd.DataFrame,
                                              price_column: str = 'Close',
                                              spans: List[int] = [12, 26]) -> pd.DataFrame:
        """
        Calculate exponential moving averages.

        Args:
            df: DataFrame with market data
            price_column: Column to calculate EMA from
            spans: List of span sizes

        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        for span in spans:
            df[f'ema_{span}'] = df[price_column].ewm(span=span, adjust=False).mean()
        return df

    def calculate_rsi(self, df: pd.DataFrame, price_column: str = 'Close',
                     period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with market data
            price_column: Column to calculate RSI from
            period: RSI period

        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(self, df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with market data
            price_column: Column to calculate MACD from

        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()
        ema_12 = df[price_column].ewm(span=12, adjust=False).mean()
        ema_26 = df[price_column].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, price_column: str = 'Close',
                                  window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with market data
            price_column: Column to calculate BB from
            window: Rolling window size
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands columns added
        """
        df = df.copy()
        df['bb_middle'] = df[price_column].rolling(window=window).mean()
        std = df[price_column].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (std * num_std)
        df['bb_lower'] = df['bb_middle'] - (std * num_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        return df

    def calculate_volatility(self, df: pd.DataFrame,
                            returns_column: str = 'returns',
                            window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility.

        Args:
            df: DataFrame with market data
            returns_column: Column containing returns
            window: Rolling window size

        Returns:
            DataFrame with volatility column added
        """
        df = df.copy()
        df['volatility'] = df[returns_column].rolling(window=window).std()
        return df

    def preprocess_market_data(self, df: pd.DataFrame, ticker_column: str = 'Ticker') -> pd.DataFrame:
        """
        Full market data preprocessing pipeline with all technical indicators.

        Args:
            df: Raw market data DataFrame
            ticker_column: Column name for ticker symbol

        Returns:
            DataFrame with all technical indicators
        """
        df = df.copy()

        # Process each ticker separately to maintain correct calculations
        if ticker_column in df.columns:
            processed_dfs = []
            for ticker in df[ticker_column].unique():
                ticker_df = df[df[ticker_column] == ticker].copy()
                ticker_df = self._add_all_indicators(ticker_df)
                processed_dfs.append(ticker_df)
            df = pd.concat(processed_dfs, ignore_index=True)
        else:
            df = self._add_all_indicators(df)

        return df

    def _add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame.

        Args:
            df: Market data for a single ticker

        Returns:
            DataFrame with all indicators added
        """
        df = self.calculate_returns(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_exponential_moving_averages(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_volatility(df)
        return df
