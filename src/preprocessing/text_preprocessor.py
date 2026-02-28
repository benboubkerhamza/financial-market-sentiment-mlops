"""
Text Preprocessing Module

Handles text cleaning, normalization, tokenization, and lemmatization
for NLP tasks on financial news and sentiment data.
"""

import pandas as pd
import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Preprocesses text data for NLP tasks.
    Handles cleaning, normalization, tokenization, and lemmatization.
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize text preprocessor.

        Args:
            remove_stopwords: Whether to remove English stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to string
        text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        tokens = word_tokenize(text)

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Apply lemmatization if enabled
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean and tokenize text.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text as string
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str,
                            output_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name for output column

        Returns:
            DataFrame with processed text column added
        """
        df = df.copy()
        df[output_column] = df[text_column].apply(self.preprocess)
        df[f'{output_column}_length'] = df[output_column].apply(lambda x: len(x.split()))
        return df
