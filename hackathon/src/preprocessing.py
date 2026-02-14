"""
Text preprocessing utilities for ESG classification
"""
import re
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from sklearn.model_selection import train_test_split

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """
    Clean text data by removing special characters, extra spaces, etc.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep periods and commas for sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Preprocess entire dataframe
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        
    Returns:
        Dataframe with cleaned text
    """
    df = df.copy()
    df[text_column] = df[text_column].apply(clean_text)
    
    # Remove empty texts
    df = df[df[text_column].str.len() > 0].reset_index(drop=True)
    
    return df


def stratified_split_multilabel(
    df: pd.DataFrame,
    label_columns: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split for multi-label data.
    Uses iterative stratification to maintain label distribution.
    
    Args:
        df: Input dataframe
        label_columns: List of label column names
        test_size: Proportion of validation set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df)
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        
        X = df.drop(columns=label_columns)
        y = df[label_columns].values
        
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=random_state
        )
        
        train_idx, val_idx = next(msss.split(X, y))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        print("\nLabel distribution in train:")
        print(train_df[label_columns].sum())
        print("\nLabel distribution in validation:")
        print(val_df[label_columns].sum())
        
        return train_df, val_df
        
    except ImportError:
        print("Warning: iterative-stratification not installed. Using simple split.")
        print("Install with: pip install iterative-stratification")
        
        # Fallback to simple train_test_split
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        return train_df, val_df


def get_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> dict:
    """
    Get statistics about text lengths
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        
    Returns:
        Dictionary of statistics
    """
    text_lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'char_length': {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'max': text_lengths.max(),
            'min': text_lengths.min(),
            'std': text_lengths.std()
        },
        'word_count': {
            'mean': word_counts.mean(),
            'median': word_counts.median(),
            'max': word_counts.max(),
            'min': word_counts.min(),
            'std': word_counts.std()
        }
    }
    
    return stats


if __name__ == "__main__":
    # Test preprocessing
    test_text = "This is a TEST text with URL https://example.com and email test@example.com!!!"
    print("Original:", test_text)
    print("Cleaned:", clean_text(test_text))
