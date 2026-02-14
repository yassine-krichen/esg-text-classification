"""
Evaluation metrics and utilities for multi-label classification
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, classification_report, multilabel_confusion_matrix,
    roc_auc_score, average_precision_score
)
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    label_names: List[str] = None
) -> Dict:
    """
    Calculate comprehensive metrics for multi-label classification
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        y_pred_proba: Predicted probabilities (optional)
        label_names: Names of labels
        
    Returns:
        Dictionary of metrics
    """
    if label_names is None:
        label_names = [f'Label_{i}' for i in range(y_true.shape[1])]
    
    metrics = {}
    
    # Overall metrics
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # Micro/Macro/Weighted averages
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'{avg}_precision'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'{avg}_recall'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'{avg}_f1'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, label in enumerate(label_names):
        metrics[f'{label}_precision'] = per_class_precision[i]
        metrics[f'{label}_recall'] = per_class_recall[i]
        metrics[f'{label}_f1'] = per_class_f1[i]
    
    # AUC metrics if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['macro_auc'] = roc_auc_score(y_true, y_pred_proba, average='macro')
            metrics['micro_auc'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            
            per_class_auc = roc_auc_score(y_true, y_pred_proba, average=None)
            for i, label in enumerate(label_names):
                metrics[f'{label}_auc'] = per_class_auc[i]
        except:
            pass  # Skip if AUC cannot be calculated
    
    return metrics


def print_metrics_report(metrics: Dict, label_names: List[str] = None):
    """
    Print formatted metrics report
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics
        label_names: Names of labels
    """
    print("=" * 70)
    print("EVALUATION METRICS REPORT")
    print("=" * 70)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS:")
    print(f"  Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Loss:                   {metrics['hamming_loss']:.4f}")
    
    # Averaged metrics
    print("\n📈 AVERAGED METRICS:")
    for avg in ['micro', 'macro', 'weighted']:
        print(f"\n  {avg.upper()} Average:")
        print(f"    Precision: {metrics[f'{avg}_precision']:.4f}")
        print(f"    Recall:    {metrics[f'{avg}_recall']:.4f}")
        print(f"    F1-Score:  {metrics[f'{avg}_f1']:.4f}")
    
    # Per-class metrics
    if label_names:
        print("\n🎯 PER-CLASS METRICS:")
        print(f"{'Label':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 50)
        for label in label_names:
            prec = metrics.get(f'{label}_precision', 0)
            rec = metrics.get(f'{label}_recall', 0)
            f1 = metrics.get(f'{label}_f1', 0)
            print(f"{label:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    # AUC metrics if available
    if 'macro_auc' in metrics:
        print("\n📉 AUC SCORES:")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"  Micro AUC: {metrics['micro_auc']:.4f}")
        
        if label_names:
            print("\n  Per-class AUC:")
            for label in label_names:
                auc = metrics.get(f'{label}_auc', 0)
                print(f"    {label}: {auc:.4f}")
    
    print("=" * 70)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    figsize: tuple = (15, 4)
):
    """
    Plot confusion matrix for each label
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Names of labels
        figsize: Figure size
    """
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, len(label_names), figsize=figsize)
    
    for i, (ax, label) in enumerate(zip(axes, label_names)):
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{label}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()


def plot_label_distribution(
    y_true: np.ndarray,
    label_names: List[str],
    title: str = "Label Distribution"
):
    """
    Plot distribution of labels
    
    Args:
        y_true: Ground truth labels
        label_names: Names of labels
        title: Plot title
    """
    counts = y_true.sum(axis=0)
    percentages = (counts / len(y_true)) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_names, counts)
    
    # Color bars
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(title)
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        plt.text(i, count, f'{int(count)}\n({pct:.1f}%)', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: List[str] = None,
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    Find optimal threshold for each class to maximize specified metric
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        label_names: Names of labels
        metric: Metric to optimize ('f1', 'precision', or 'recall')
        
    Returns:
        Dictionary of optimal thresholds per class
    """
    if label_names is None:
        label_names = [f'Label_{i}' for i in range(y_true.shape[1])]
    
    optimal_thresholds = {}
    
    for i, label in enumerate(label_names):
        best_threshold = 0.5
        best_score = 0
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba[:, i] >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[:, i], y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true[:, i], y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true[:, i], y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[label] = best_threshold
        print(f"{label}: threshold={best_threshold:.2f}, {metric}={best_score:.4f}")
    
    return optimal_thresholds


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    y_true = np.random.randint(0, 2, (100, 4))
    y_pred = np.random.randint(0, 2, (100, 4))
    y_pred_proba = np.random.rand(100, 4)
    
    label_names = ['E', 'S', 'G', 'non_ESG']
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba, label_names)
    print_metrics_report(metrics, label_names)
