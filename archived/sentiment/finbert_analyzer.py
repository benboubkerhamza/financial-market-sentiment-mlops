"""
FinBERT Sentiment Analyzer

Implementation using FinBERT - a BERT model pre-trained on financial text.
Specialized for financial sentiment analysis with better accuracy on financial news.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .sentiment_analyzer import SentimentAnalyzer

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False


class FinBERTSentimentAnalyzer(SentimentAnalyzer):
    """
    FinBERT-based sentiment analyzer.
    Uses pre-trained FinBERT model from HuggingFace for financial sentiment analysis.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        
        if not FINBERT_AVAILABLE:
            raise ImportError(
                "FinBERT dependencies not available. Install with: "
                "pip install transformers torch"
            )
        
        print(f"Loading FinBERT model: {model_name}...")
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (FinBERT outputs: 0=negative, 1=neutral, 2=positive)
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        print(f"✓ FinBERT model loaded on {self.device}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores:
            - positive: positive sentiment probability (0-1)  
            - negative: negative sentiment probability (0-1)
            - neutral: neutral sentiment probability (0-1)
            - label: predicted sentiment label
        """
        if not text or pd.isna(text):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'neutral'
            }
        
        # Tokenize
        inputs = self.tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()[0]
        
        return {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2]),
            'label': self.label_map[np.argmax(probs)]
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'processed_text',
                         batch_size: int = 16) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            batch_size: Batch size for processing (default: 16)
            
        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()
        
        # Ensure text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        print(f"Analyzing sentiment for {len(df)} texts using FinBERT...")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        
        # Process in batches for efficiency
        texts = df[text_column].fillna('').astype(str).tolist()
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._analyze_batch(batch_texts)
            all_results.extend(batch_results)
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")
        
        # Extract scores into separate columns
        results_df = pd.DataFrame(all_results)
        df['finbert_positive'] = results_df['positive']
        df['finbert_negative'] = results_df['negative']
        df['finbert_neutral'] = results_df['neutral']
        df['finbert_sentiment'] = results_df['label']
        
        print(f"✓ FinBERT sentiment analysis completed")
        print(f"  Distribution: {df['finbert_sentiment'].value_counts().to_dict()}")
        
        return df
    
    def _analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convert to results
        probs = probabilities.cpu().numpy()
        results = []
        
        for prob in probs:
            results.append({
                'negative': float(prob[0]),
                'neutral': float(prob[1]),
                'positive': float(prob[2]),
                'label': self.label_map[np.argmax(prob)]
            })
        
        return results
