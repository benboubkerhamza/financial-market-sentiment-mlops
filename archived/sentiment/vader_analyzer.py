"""
VADER Sentiment Analyzer

Implementation using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Fast, rule-based sentiment analysis suitable for baseline.
"""

import pandas as pd
from typing import Dict
from .sentiment_analyzer import SentimentAnalyzer

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    # Download VADER lexicon if not present
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class VaderSentimentAnalyzer(SentimentAnalyzer):
    """
    VADER-based sentiment analyzer.
    Good baseline, especially for social media and informal text.
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        super().__init__()
        
        if not VADER_AVAILABLE:
            raise ImportError(
                "VADER not available. Install with: pip install nltk"
            )
        
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores:
            - positive: positive sentiment score (0-1)
            - negative: negative sentiment score (0-1)
            - neutral: neutral sentiment score (0-1)
            - compound: compound score (-1 to 1)
        """
        if not text or pd.isna(text):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(str(text))
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()
        
        # Ensure text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        print(f"Analyzing sentiment for {len(df)} texts using VADER...")
        
        # Analyze each text
        sentiment_results = df[text_column].apply(self.analyze_text)
        
        # Extract scores into separate columns
        df['vader_positive'] = sentiment_results.apply(lambda x: x['positive'])
        df['vader_negative'] = sentiment_results.apply(lambda x: x['negative'])
        df['vader_neutral'] = sentiment_results.apply(lambda x: x['neutral'])
        df['vader_compound'] = sentiment_results.apply(lambda x: x['compound'])
        
        # Determine dominant sentiment using compound score
        df['vader_sentiment'] = df['vader_compound'].apply(self._compound_to_label)
        
        print(f"✓ VADER sentiment analysis completed")
        print(f"  Distribution: {df['vader_sentiment'].value_counts().to_dict()}")
        
        return df
    
    def _compound_to_label(self, compound_score: float) -> str:
        """
        Convert VADER compound score to sentiment label.
        
        Args:
            compound_score: VADER compound score (-1 to 1)
            
        Returns:
            'positive', 'negative', or 'neutral'
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
