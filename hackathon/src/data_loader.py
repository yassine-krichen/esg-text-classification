"""
PyTorch Dataset and DataLoader for BERT-based ESG classification

This module creates custom dataset classes that:
1. Tokenize text using BERT tokenizer
2. Handle multi-label targets
3. Create proper attention masks
4. Batch data efficiently
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class ESGDataset(Dataset):
    """
    Custom PyTorch Dataset for ESG text classification
    
    This is similar to how we handled image data in the course,
    but adapted for text with BERT tokenization.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray = None,
        tokenizer: BertTokenizer = None,
        max_length: int = 512
    ):
        """
        Args:
            texts: List of text strings
            labels: Array of shape (n_samples, n_labels) for multi-label classification
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length (BERT supports up to 512)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Training mode if labels provided
        self.is_test = labels is None
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset
        
        Similar to how we accessed images in the course,
        but here we return tokenized text instead.
        """
        text = str(self.texts[idx])
        
        # Tokenize the text
        # This is like preprocessing images, but for text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',     # Pad to max_length
            truncation=True,          # Truncate if longer than max_length
            return_attention_mask=True,  # Create attention mask
            return_tensors='pt'       # Return PyTorch tensors
        )
        
        # Extract the tensors and squeeze the batch dimension
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        if self.is_test:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # Convert labels to tensor
            labels = torch.FloatTensor(self.labels[idx])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    tokenizer: BertTokenizer = None,
    max_length: int = 512,
    batch_size: int = 16,
    label_columns: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and test sets
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe (optional)
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for training
        label_columns: List of label column names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if label_columns is None:
        label_columns = ['E', 'S', 'G', 'non_ESG']
    
    # Create datasets
    train_dataset = ESGDataset(
        texts=train_df['text'].tolist(),
        labels=train_df[label_columns].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = ESGDataset(
        texts=val_df['text'].tolist(),
        labels=val_df[label_columns].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=0
    )
    
    # Test loader (if test data provided)
    test_loader = None
    if test_df is not None:
        test_dataset = ESGDataset(
            texts=test_df['text'].tolist(),
            labels=None,  # No labels for test set
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing ESGDataset...")
    
    from transformers import BertTokenizer
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Sample data
    texts = [
        "The company reduced carbon emissions by 20%.",
        "We improved employee diversity and inclusion programs.",
        "The board of directors consists of 8 members."
    ]
    labels = np.array([
        [1, 0, 0, 0],  # E
        [0, 1, 0, 0],  # S
        [0, 0, 1, 0]   # G
    ])
    
    # Create dataset
    dataset = ESGDataset(texts, labels, tokenizer, max_length=128)
    
    # Get first sample
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Labels: {sample['labels']}")
    
    # Decode to see what BERT sees
    decoded = tokenizer.decode(sample['input_ids'])
    print(f"\nOriginal text: {texts[0]}")
    print(f"Tokenized text: {decoded}")
    
    print("\n✅ ESGDataset test passed!")
