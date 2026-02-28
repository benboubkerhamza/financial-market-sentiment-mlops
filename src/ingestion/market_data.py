"""
Market Data Fetching Module

Handles fetching market data from yfinance API
"""

import pandas as pd
import yfinance as yf
from typing import List
from .base import BaseIngestion


class MarketDataFetcher(BaseIngestion):
    """Class to handle market data fetching from yfinance"""
    
    def fetch_market_data(
        self, 
        tickers: List[str],
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
        self.logger.info(f"Fetching market data for tickers: {tickers}")
        
        market_data = []
        
        for ticker in tickers:
            try:
                self.logger.info(f"Downloading data for {ticker}")
                stock = yf.Ticker(ticker)
                
                if start_date and end_date:
                    df = stock.history(start=start_date, end=end_date)
                else:
                    df = stock.history(period=period)
                
                if not df.empty:
                    df['Ticker'] = ticker
                    df.reset_index(inplace=True)
                    market_data.append(df)
                    self.logger.info(f"Retrieved {len(df)} records for {ticker}")
                else:
                    self.logger.warning(f"No data retrieved for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        if market_data:
            combined_df = pd.concat(market_data, ignore_index=True)
            
            # Save to processed directory
            output_path = self.processed_dir / "market_data_raw.csv"
            combined_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(combined_df)} market records to {output_path}")
            
            return combined_df
        else:
            self.logger.warning("No market data retrieved for any ticker")
            return pd.DataFrame()
    
    def fetch_sp500_tickers(self) -> List[str]:
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
            self.logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            # Return some common tickers as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    
    def get_company_info(self, ticker: str) -> dict:
        """
        Get detailed company information
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            self.logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {}
