"""
Preprocessing module for financial news sentiment MLOps pipeline.

Reads raw CSV files from the `data/` directory, cleans and tokenises
the text, engineers features, and writes processed datasets ready for
model training.
"""

import os
import re
import glob
import logging

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download required NLTK resources on first run
# "omw-1.4" is the Open Multilingual Wordnet v1.4, required by WordNetLemmatizer
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LABEL_MAP = {"Bearish": 0, "Somewhat-Bearish": 1, "Neutral": 2, "Somewhat-Bullish": 3, "Bullish": 4}

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lower-case, remove HTML/URLs/punctuation, lemmatize and strip stop-words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)               # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in text.split()
        if tok not in _stop_words and len(tok) > 2
    ]
    return " ".join(tokens)


def load_latest_raw(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the most recently ingested raw CSV file."""
    pattern = os.path.join(data_dir, "raw_news_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")
    latest = files[-1]
    logger.info("Loading raw data from %s", latest)
    return pd.read_csv(latest)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the overall_sentiment_label column to integer classes.

    Falls back to LabelEncoder when the label is not in the predefined
    LABEL_MAP (e.g., data sourced from NewsAPI without pre-labelled sentiment).
    """
    df = df.copy()
    if "overall_sentiment_label" not in df.columns:
        raise ValueError("Column 'overall_sentiment_label' missing from DataFrame.")

    # Drop rows with empty labels
    df = df[df["overall_sentiment_label"].notna() & (df["overall_sentiment_label"] != "")]

    if df["overall_sentiment_label"].isin(LABEL_MAP.keys()).all():
        df["label"] = df["overall_sentiment_label"].map(LABEL_MAP)
    else:
        logger.warning("Unexpected label values found; falling back to LabelEncoder.")
        enc = LabelEncoder()
        df["label"] = enc.fit_transform(df["overall_sentiment_label"])

    return df


def build_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title + summary into a single cleaned text feature column."""
    df = df.copy()
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()
    df["text_clean"] = df["text"].apply(clean_text)
    return df


def preprocess(data_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Returns:
        Dictionary with keys ``train``, ``val``, ``test`` as DataFrames.
    """
    df = load_latest_raw(data_dir)
    logger.info("Loaded %d raw records.", len(df))

    df = build_text_feature(df)
    df = encode_labels(df)

    # Keep only required columns
    df = df[["ticker", "text_clean", "label", "published_at"]].dropna()
    logger.info("%d records after cleaning.", len(df))

    # Stratified split: 70 / 15 / 15
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(data_dir, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        logger.info("Saved %s split (%d records) to %s", split_name, len(split_df), out_path)

    return {"train": train_df, "val": val_df, "test": test_df}


if __name__ == "__main__":
    preprocess()
