"""
Base Sentiment Analyzer Interface

Defines the interface for all sentiment analyzers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Union


class SentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        pass
    
    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        pass
    
    @abstractmethod
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with added sentiment columns
        """
        pass
    
    def get_dominant_sentiment(self, scores: Dict[str, float]) -> str:
        """
        Get the dominant sentiment label from scores.
        
        Args:
            scores: Dictionary of sentiment scores
            
        Returns:
            'positive', 'negative', or 'neutral'
        """
        if 'positive' in scores and 'negative' in scores:
            if scores['positive'] > scores['negative']:
                # Check if positive is significant enough
                if scores['positive'] > 0.1:
                    return 'positive'
                else:
                    return 'neutral'
            elif scores['negative'] > scores['positive']:
                # Check if negative is significant enough
                if scores['negative'] > 0.1:
                    return 'negative'
                else:
                    return 'neutral'
            else:
                return 'neutral'
        
        # Fallback
        return 'neutral'
