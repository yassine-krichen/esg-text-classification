"""
Training utilities for BERT model

Includes functions for:
- Training loop
- Validation loop
- Learning rate scheduling
- Model checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import time


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    loss_fn,
    device: str,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Similar to the training loop in utils.py from the course,
    but adapted for BERT and multi-label classification.
    
    Args:
        model: BERT classifier
        train_loader: Training data loader
        optimizer: Optimizer (AdamW recommended for BERT)
        scheduler: Learning rate scheduler
        loss_fn: Loss function (BCEWithLogitsLoss for multi-label)
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate accuracy (threshold at 0.5)
        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.numel()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = correct_predictions / total_predictions
    
    return avg_loss, avg_accuracy


def validate_epoch(
    model: nn.Module,
    val_loader,
    loss_fn,
    device: str,
    epoch: int
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate for one epoch
    
    Args:
        model: BERT classifier
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device
        epoch: Current epoch number
        
    Returns:
        Tuple of (avg_loss, avg_accuracy, all_predictions, all_labels)
    """
    model.eval()
    
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
    
    with torch.no_grad():  # No gradient calculation during validation
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            
            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store for metrics calculation
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = correct_predictions / total_predictions
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_probabilities = np.vstack(all_probabilities)
    
    return avg_loss, avg_accuracy, all_predictions, all_labels, all_probabilities


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int,
    learning_rate: float = 2e-5,
    device: str = 'cuda',
    save_path: str = None,
    label_names: list = None
) -> Dict:
    """
    Complete training loop with validation
    
    Args:
        model: BERT classifier
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save best model
        label_names: Names of labels for evaluation
        
    Returns:
        Dictionary with training history
    """
    if label_names is None:
        label_names = ['E', 'S', 'G', 'non_ESG']
    
    # Loss function - Binary Cross Entropy with Logits Loss
    # This is like the loss we used in the course, but for multi-label
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Optimizer - AdamW is recommended for BERT
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'epochs': []
    }
    
    best_val_f1 = 0.0
    
    print(f"🚀 Starting training...")
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup steps: {warmup_steps} / {total_steps}")
    print("=" * 70)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
            model, val_loader, loss_fn, device, epoch
        )
        
        # Calculate detailed metrics
        from src.evaluation import calculate_metrics
        metrics = calculate_metrics(
            val_labels, val_preds, val_probs, label_names
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(metrics['macro_f1'])
        history['epochs'].append(epoch + 1)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\n📊 Epoch {epoch+1}/{n_epochs} Summary ({epoch_time:.1f}s)")
        print(f"{'─' * 70}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Per-class F1: E={metrics['E_f1']:.3f}, S={metrics['S_f1']:.3f}, "
              f"G={metrics['G_f1']:.3f}, non_ESG={metrics['non_ESG_f1']:.3f}")
        
        # Save best model
        if metrics['macro_f1'] > best_val_f1:
            best_val_f1 = metrics['macro_f1']
            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'history': history
                }, save_path)
                print(f"💾 Saved best model (F1: {best_val_f1:.4f}) to {save_path}")
        
        print("=" * 70)
    
    print(f"\n✅ Training complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    return history


def predict(
    model: nn.Module,
    data_loader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    print("🔮 Making predictions...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get probabilities
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    
    return all_predictions, all_probabilities


if __name__ == "__main__":
    print("✅ Training utilities module loaded successfully!")
    print("Functions available:")
    print("  - train_epoch()")
    print("  - validate_epoch()")
    print("  - train_model()")
    print("  - predict()")
