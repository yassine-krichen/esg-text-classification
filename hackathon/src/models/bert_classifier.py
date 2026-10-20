"""
BERT-based classifier for ESG multi-label classification

This model is similar to the VGG16 transfer learning we did in the course,
but for text instead of images.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict


class BERTClassifier(nn.Module):
    """
    BERT-based multi-label classifier
    
    Architecture:
    1. BERT base model (pretrained) - extracts text features
    2. Dropout layer - prevents overfitting
    3. Linear classifier - maps to 4 labels
    
    This is like using VGG16 features + custom classifier,
    but for text with BERT embeddings.
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        bert_model_name: str = 'bert-base-uncased',
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Args:
            n_classes: Number of output classes (4 for E, S, G, non_ESG)
            bert_model_name: Name of pretrained BERT model
            dropout: Dropout probability
            freeze_bert: If True, freeze BERT weights (faster training)
        """
        super(BERTClassifier, self).__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if specified
        # Similar to freezing VGG16 base in the course
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT hidden size (768 for bert-base)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        # Similar to the custom layers we added to VGG16
        self.classifier = nn.Linear(self.bert_hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, max_length)
            attention_mask: Attention mask (batch_size, max_length)
            
        Returns:
            Logits for each class (batch_size, n_classes)
        """
        # Get BERT outputs
        # The [CLS] token (first token) represents the whole sequence
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation
        # This is like the global average pooling in CNNs
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits (before sigmoid)
        logits = self.classifier(pooled_output)  # (batch_size, n_classes)
        
        return logits


class BERTClassifierWithLSTM(nn.Module):
    """
    Advanced: BERT + LSTM classifier
    
    This adds an LSTM layer on top of BERT to better capture
    sequential patterns. More complex but potentially better.
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        bert_model_name: str = 'bert-base-uncased',
        lstm_hidden_size: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super(BERTClassifierWithLSTM, self).__init__()
        
        # BERT base
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.bert_hidden_size,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier (input is 2 * lstm_hidden_size due to bidirectional)
        self.classifier = nn.Linear(2 * lstm_hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with LSTM
        """
        # Get BERT sequence outputs (not just [CLS])
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get all token representations
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        
        # Use the last hidden state from both directions
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Get logits
        logits = self.classifier(hidden)
        
        return logits


def get_model(
    model_type: str = 'bert',
    n_classes: int = 4,
    bert_model_name: str = 'bert-base-uncased',
    dropout: float = 0.3,
    freeze_bert: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create and initialize model
    
    Args:
        model_type: 'bert' or 'bert-lstm'
        n_classes: Number of output classes
        bert_model_name: Pretrained model name
        dropout: Dropout probability
        freeze_bert: Whether to freeze BERT weights
        device: Device to load model on
        
    Returns:
        Initialized model on specified device
    """
    if model_type == 'bert':
        model = BERTClassifier(
            n_classes=n_classes,
            bert_model_name=bert_model_name,
            dropout=dropout,
            freeze_bert=freeze_bert
        )
    elif model_type == 'bert-lstm':
        model = BERTClassifierWithLSTM(
            n_classes=n_classes,
            bert_model_name=bert_model_name,
            dropout=dropout,
            freeze_bert=freeze_bert
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing BERTClassifier...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = get_model(
        model_type='bert',
        n_classes=4,
        freeze_bert=False,
        device=device
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected: ({batch_size}, 4)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    print("\n✅ BERTClassifier test passed!")
