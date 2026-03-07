"""
Apply Sentiment Analysis

This script applies VADER and FinBERT sentiment analyzers to news data.
It also validates the models on the training dataset.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.sentiment.vader_analyzer import VaderSentimentAnalyzer
from src.sentiment.finbert_analyzer import FinBERTSentimentAnalyzer
from sklearn.metrics import classification_report, confusion_matrix


def load_data():
    """Load processed datasets"""
    data_dir = project_root / 'data' / 'processed'
    
    print("Loading datasets...")
    
    # Load training data (for validation)
    sentiment_training_path = data_dir / 'sentiment_processed.csv'
    sentiment_training_df = pd.read_csv(sentiment_training_path)
    print(f"  Loaded {len(sentiment_training_df)} training samples")
    
    # Load news data (to apply sentiment)
    news_path = data_dir / 'news_processed.csv'
    news_df = pd.read_csv(news_path)
    print(f"  Loaded {len(news_df)} news articles")
    
    return sentiment_training_df, news_df


def validate_on_training_data(analyzer, training_df, analyzer_name, 
                              sentiment_col_name):
    """
    Validate sentiment analyzer on training data.
    
    Args:
        analyzer: Sentiment analyzer instance
        training_df: Training DataFrame with 'sentiment' ground truth
        analyzer_name: Name for display (e.g., 'VADER', 'FinBERT')
        sentiment_col_name: Column name for sentiment label in results
    """
    print(f"\n{'='*60}")
    print(f"Validating {analyzer_name} on Training Data")
    print(f"{'='*60}")
    
    # Apply analyzer
    results_df = analyzer.analyze_dataframe(training_df, text_column='processed_text')
    
    # Compare predictions with ground truth
    y_true = results_df['sentiment']
    y_pred = results_df[sentiment_col_name]
    
    # Print classification report
    print(f"\nClassification Report for {analyzer_name}:")
    print(classification_report(y_true, y_pred, 
                               labels=['positive', 'neutral', 'negative']))
    
    # Print confusion matrix
    print(f"\nConfusion Matrix for {analyzer_name}:")
    cm = confusion_matrix(y_true, y_pred, 
                         labels=['positive', 'neutral', 'negative'])
    cm_df = pd.DataFrame(cm, 
                        index=['True Positive', 'True Neutral', 'True Negative'],
                        columns=['Pred Positive', 'Pred Neutral', 'Pred Negative'])
    print(cm_df)
    
    # Calculate accuracy by class
    print(f"\nAccuracy by Class:")
    for label in ['positive', 'neutral', 'negative']:
        mask = y_true == label
        if mask.sum() > 0:
            accuracy = (y_true[mask] == y_pred[mask]).sum() / mask.sum()
            print(f"  {label.capitalize()}: {accuracy:.2%}")
    
    return results_df


def apply_sentiment_to_news(news_df, use_finbert=True):
    """
    Apply sentiment analysis to news articles.
    
    Args:
        news_df: News DataFrame
        use_finbert: Whether to use FinBERT (default: True)
        
    Returns:
        DataFrame with sentiment scores
    """
    print(f"\n{'='*60}")
    print(f"Applying Sentiment Analysis to News Articles")
    print(f"{'='*60}")
    
    # Always apply VADER (fast baseline)
    print("\n[1/2] Applying VADER...")
    vader = VaderSentimentAnalyzer()
    news_df = vader.analyze_dataframe(news_df, text_column='processed_text')
    
    # Optionally apply FinBERT (slow but accurate)
    if use_finbert:
        print("\n[2/2] Applying FinBERT...")
        try:
            finbert = FinBERTSentimentAnalyzer()
            news_df = finbert.analyze_dataframe(news_df, text_column='processed_text',
                                               batch_size=8)
        except Exception as e:
            print(f"⚠️  FinBERT failed: {e}")
            print("  Continuing with VADER only...")
    else:
        print("\n[2/2] Skipping FinBERT (use_finbert=False)")
    
    return news_df


def main():
    """Main execution"""
    print("="*60)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Load data
    training_df, news_df = load_data()
    
    # Validate VADER on training data
    print("\n" + "="*60)
    print("STEP 1: VALIDATE VADER ON TRAINING DATA")
    print("="*60)
    vader = VaderSentimentAnalyzer()
    validate_on_training_data(vader, training_df, "VADER", "vader_sentiment")
    
    # Validate FinBERT on training data (sample for speed)
    print("\n" + "="*60)
    print("STEP 2: VALIDATE FinBERT ON TRAINING DATA (500 samples)")
    print("="*60)
    
    try:
        finbert = FinBERTSentimentAnalyzer()
        
        # Sample 500 balanced samples for validation (faster)
        sample_size = min(500, len(training_df))
        training_sample = training_df.groupby('sentiment', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 3), random_state=42)
        )
        
        validate_on_training_data(finbert, training_sample, "FinBERT", 
                                 "finbert_sentiment")
    except Exception as e:
        print(f"⚠️  FinBERT validation failed: {e}")
        print("  Skipping FinBERT validation...")
    
    # Apply sentiment to news data
    print("\n" + "="*60)
    print("STEP 3: APPLY SENTIMENT TO NEWS ARTICLES")
    print("="*60)
    
    news_with_sentiment = apply_sentiment_to_news(news_df, use_finbert=True)
    
    # Save results
    output_path = project_root / 'data' / 'processed' / 'news_with_sentiment.csv'
    news_with_sentiment.to_csv(output_path, index=False)
    print(f"\n✓ Saved news with sentiment: {output_path}")
    print(f"  Total articles: {len(news_with_sentiment)}")
    print(f"  Columns: {list(news_with_sentiment.columns)}")
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print("SENTIMENT SUMMARY")
    print(f"{'='*60}")
    
    print("\nVADER Sentiment Distribution:")
    print(news_with_sentiment['vader_sentiment'].value_counts())
    print(f"\nVADER Compound Score Stats:")
    print(news_with_sentiment['vader_compound'].describe())
    
    if 'finbert_sentiment' in news_with_sentiment.columns:
        print("\nFinBERT Sentiment Distribution:")
        print(news_with_sentiment['finbert_sentiment'].value_counts())
        print(f"\nFinBERT Confidence (max prob):")
        max_prob = news_with_sentiment[['finbert_positive', 'finbert_neutral', 
                                        'finbert_negative']].max(axis=1)
        print(max_prob.describe())
    
    print("\n" + "="*60)
    print("✓ SENTIMENT ANALYSIS PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
