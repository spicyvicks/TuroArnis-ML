"""
Step 4d: Evaluate Model (Enhanced with Research-Quality Visualizations)

Evaluates a trained Hybrid GCN model on the test set.
Generates:
- Confusion Matrix
- Classification Report
- Per-Class Accuracy
- Threshold Sensitivity Analysis (Line Plots)
- Cross-Threshold Analysis (Heatmaps)
- Stacked Bar Charts with Metrics
- Ground Truth Distribution

Usage:
    python hybrid_classifier/4d_evaluate_model.py --viewpoint front
    python hybrid_classifier/4d_evaluate_model.py --viewpoint left
    python hybrid_classifier/4d_evaluate_model.py --viewpoint right
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

# Import from training script
import sys
import importlib.util

# Dynamically import module that starts with a number
script_dir = Path(__file__).resolve().parent
module_path = script_dir / "4c_train_hybrid_gcn_v2.py"

spec = importlib.util.spec_from_file_location("train_module", module_path)
train_module = importlib.util.module_from_spec(spec)
sys.modules["train_module"] = train_module
spec.loader.exec_module(train_module)

HybridGCN = train_module.HybridGCN
load_hybrid_graph_data = train_module.load_hybrid_graph_data
CLASS_NAMES = train_module.CLASS_NAMES

# Config
MODEL_DIR = Path("hybrid_classifier/models")
OUTPUT_DIR = Path("hybrid_classifier/evaluation")

def plot_threshold_sensitivity(all_probs, all_labels, viewpoint):
    """Plot accuracy and kappa vs confidence threshold (inspired by research papers)"""
    thresholds = np.arange(0.0, 1.0, 0.05)
    accuracies = []
    kappas = []
    sample_counts = []
    
    for thresh in thresholds:
        # Get predictions above threshold
        max_probs = np.max(all_probs, axis=1)
        valid_mask = max_probs >= thresh
        
        if valid_mask.sum() == 0:
            accuracies.append(0)
            kappas.append(0)
            sample_counts.append(0)
            continue
            
        valid_preds = np.argmax(all_probs[valid_mask], axis=1)
        valid_labels = all_labels[valid_mask]
        
        acc = accuracy_score(valid_labels, valid_preds)
        kappa = cohen_kappa_score(valid_labels, valid_preds)
        
        accuracies.append(acc)
        kappas.append(kappa)
        sample_counts.append(valid_mask.sum())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs Threshold
    ax1.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=6, label='Accuracy')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Accuracy vs Threshold - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Kappa vs Threshold
    ax2.plot(thresholds, kappas, marker='s', linewidth=2, markersize=6, color='orange', label="Cohen's Kappa")
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel("Cohen's Kappa", fontsize=12)
    ax2.set_title(f"Kappa vs Threshold - {viewpoint.upper()}", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"threshold_sensitivity_{viewpoint}.png", dpi=150)
    print(f"Threshold sensitivity plot saved to {OUTPUT_DIR / f'threshold_sensitivity_{viewpoint}.png'}")
    plt.close()


def plot_cross_threshold_heatmap(all_probs, all_labels, viewpoint):
    """Plot accuracy and kappa heatmaps across dual thresholds"""
    thresholds = np.arange(0.5, 1.55, 0.15)  # Realistic thresholds for pose estimation
    
    # For dual threshold, we'll use: max_prob >= T1 AND second_max_prob <= T2
    # This simulates the "spread ratio" concept from the paper
    acc_matrix = np.zeros((len(thresholds), len(thresholds)))
    kappa_matrix = np.zeros((len(thresholds), len(thresholds)))
    
    sorted_probs = np.sort(all_probs, axis=1)
    max_probs = sorted_probs[:, -1]
    second_max_probs = sorted_probs[:, -2]
    spread_ratios = max_probs / (second_max_probs + 1e-8)  # Avoid division by zero
    
    for i, t1 in enumerate(thresholds):
        for j, t2 in enumerate(thresholds):
            # Filter by spread ratio
            valid_mask = spread_ratios >= t1
            
            if valid_mask.sum() == 0:
                continue
                
            valid_preds = np.argmax(all_probs[valid_mask], axis=1)
            valid_labels = all_labels[valid_mask]
            
            acc_matrix[i, j] = accuracy_score(valid_labels, valid_preds)
            kappa_matrix[i, j] = cohen_kappa_score(valid_labels, valid_preds)
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy heatmap
    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=[f'{t:.2f}' for t in thresholds],
                yticklabels=[f'{t:.2f}' for t in thresholds],
                vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Accuracy'})
    ax1.set_xlabel('Prediction Threshold (T_pred)', fontsize=12)
    ax1.set_ylabel('Spread Ratio Threshold (T_spread)', fontsize=12)
    ax1.set_title(f'Accuracy Heatmap - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    
    # Kappa heatmap
    sns.heatmap(kappa_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[f'{t:.2f}' for t in thresholds],
                yticklabels=[f'{t:.2f}' for t in thresholds],
                vmin=0, vmax=1, ax=ax2, cbar_kws={'label': "Cohen's Kappa"})
    ax2.set_xlabel('Prediction Threshold (T_pred)', fontsize=12)
    ax2.set_ylabel('Spread Ratio Threshold (T_spread)', fontsize=12)
    ax2.set_title(f"Cohen's Kappa Heatmap - {viewpoint.upper()}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cross_threshold_heatmap_{viewpoint}.png", dpi=150)
    print(f"Cross-threshold heatmap saved to {OUTPUT_DIR / f'cross_threshold_heatmap_{viewpoint}.png'}")
    plt.close()


def plot_stacked_distribution(all_probs, all_labels, viewpoint):
    """Plot stacked bar chart showing class distribution across thresholds with accuracy overlay"""
    thresholds = np.arange(0.5, 1.55, 0.15)  # Realistic thresholds for pose estimation
    
    # Prepare data
    class_counts = {i: [] for i in range(len(CLASS_NAMES))}
    accuracies = []
    kappas = []
    
    sorted_probs = np.sort(all_probs, axis=1)
    max_probs = sorted_probs[:, -1]
    second_max_probs = sorted_probs[:, -2]
    spread_ratios = max_probs / (second_max_probs + 1e-8)
    
    for thresh in thresholds:
        valid_mask = spread_ratios >= thresh
        
        if valid_mask.sum() == 0:
            for i in range(len(CLASS_NAMES)):
                class_counts[i].append(0)
            accuracies.append(0)
            kappas.append(0)
            continue
        
        valid_preds = np.argmax(all_probs[valid_mask], axis=1)
        valid_labels = all_labels[valid_mask]
        
        # Count predictions per class
        for i in range(len(CLASS_NAMES)):
            class_counts[i].append(np.sum(valid_preds == i))
        
        accuracies.append(accuracy_score(valid_labels, valid_preds) * 100)
        kappas.append(cohen_kappa_score(valid_labels, valid_preds) * 100)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(thresholds))
    bottom = np.zeros(len(thresholds))
    
    # Use a colormap for classes
    colors = plt.cm.tab20(np.linspace(0, 1, len(CLASS_NAMES)))
    
    for i in range(len(CLASS_NAMES)):
        counts = class_counts[i]
        ax.bar(x, counts, bottom=bottom, label=CLASS_NAMES[i], color=colors[i], width=0.6)
        
        # Add count labels in the middle of each segment
        for j, count in enumerate(counts):
            if count > 0:
                ax.text(j, bottom[j] + count/2, str(int(count)), 
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        bottom += counts
    
    # Overlay accuracy and kappa lines
    ax2 = ax.twinx()
    ax2.plot(x, accuracies, marker='o', linewidth=2.5, markersize=8, 
             color='black', label='Accuracy (%)', linestyle='-')
    ax2.plot(x, kappas, marker='s', linewidth=2.5, markersize=8,
             color='darkred', label="Kappa x100", linestyle='--')
    ax2.set_ylabel('Accuracy / Kappa (%)', fontsize=12)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Spread Ratio Threshold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'Class Distribution vs Threshold - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"stacked_distribution_{viewpoint}.png", dpi=150, bbox_inches='tight')
    print(f"Stacked distribution plot saved to {OUTPUT_DIR / f'stacked_distribution_{viewpoint}.png'}")
    plt.close()


def plot_per_class_accuracy(cm, viewpoint):
    """Plot per-class accuracy as a bar chart"""
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(class_acc)
    sorted_names = [CLASS_NAMES[i] for i in sorted_indices]
    sorted_acc = class_acc[sorted_indices]
    
    # Color code by performance level
    colors = ['#d32f2f' if acc < 0.5 else '#ff9800' if acc < 0.7 else '#4caf50' for acc in sorted_acc]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(sorted_names)), sorted_acc, color=colors)
    
    # Add percentage labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-Class Accuracy - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='50% threshold')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.3, label='70% threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"per_class_accuracy_{viewpoint}.png", dpi=150, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to {OUTPUT_DIR / f'per_class_accuracy_{viewpoint}.png'}")
    plt.close()


def plot_classification_report_heatmap(all_labels, all_preds, viewpoint):
    """Plot classification report as a heatmap (precision, recall, F1-score)"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    
    # Create matrix: [classes x metrics]
    report_matrix = np.column_stack([precision, recall, f1])
    
    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(report_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title(f'Classification Report - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Class', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"classification_report_heatmap_{viewpoint}.png", dpi=150, bbox_inches='tight')
    print(f"Classification report heatmap saved to {OUTPUT_DIR / f'classification_report_heatmap_{viewpoint}.png'}")
    plt.close()


def plot_training_history(viewpoint):
    """Plot training history (loss and accuracy curves)"""
    import json
    
    history_file = MODEL_DIR / f"history_{viewpoint}.json"
    if not history_file.exists():
        print(f"Training history not found: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], marker='o', linewidth=2, markersize=4, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], marker='o', linewidth=2, markersize=4, label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], marker='s', linewidth=2, markersize=4, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training & Test Accuracy - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # Highlight best test accuracy
    best_epoch = np.argmax(history['test_acc'])
    best_acc = history['test_acc'][best_epoch]
    ax2.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.2%}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"training_history_{viewpoint}.png", dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {OUTPUT_DIR / f'training_history_{viewpoint}.png'}")
    plt.close()


def plot_ground_truth_distribution(all_labels, viewpoint):
    """Plot ground truth class distribution"""
    thresholds = np.arange(0.5, 1.55, 0.15)  # Realistic thresholds for pose estimation
    
    # Count ground truth occurrences
    class_counts = {i: [] for i in range(len(CLASS_NAMES))}
    
    for _ in thresholds:
        # Ground truth doesn't change with threshold, but we show it for comparison
        for i in range(len(CLASS_NAMES)):
            class_counts[i].append(np.sum(all_labels == i))
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(thresholds))
    bottom = np.zeros(len(thresholds))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(CLASS_NAMES)))
    
    for i in range(len(CLASS_NAMES)):
        counts = class_counts[i]
        ax.bar(x, counts, bottom=bottom, label=CLASS_NAMES[i], color=colors[i], width=0.6)
        
        # Add count labels
        for j, count in enumerate(counts):
            if count > 0:
                ax.text(j, bottom[j] + count/2, str(int(count)),
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        bottom += counts
    
    ax.set_xlabel('Threshold (for reference)', fontsize=12)
    ax.set_ylabel('Number of Samples (N={})'.format(len(all_labels)), fontsize=12)
    ax.set_title(f'Ground Truth Distribution - {viewpoint.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ground_truth_distribution_{viewpoint}.png", dpi=150, bbox_inches='tight')
    print(f"Ground truth distribution plot saved to {OUTPUT_DIR / f'ground_truth_distribution_{viewpoint}.png'}")
    plt.close()


def evaluate_model(viewpoint, hidden_dim=256, num_layers=3, dropout=0.5):
    """Evaluate a specific viewpoint model with advanced visualizations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print(f"EVALUATING HYBRID GCN V2: {viewpoint.upper()}")
    print(f"{'='*60}\n")
    
    # 1. Load Test Data
    print("Loading test data...")
    _, test_graphs = load_hybrid_graph_data(viewpoint)
    
    if len(test_graphs) == 0:
        print("Error: No test data found!")
        return
        
    print(f"Test samples: {len(test_graphs)}")
    
    # Create dataloader
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model_path = MODEL_DIR / f"hybrid_gcn_v2_{viewpoint}.pth"
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get dimensions from checkpoint if available, else use defaults
    node_feat_dim = checkpoint.get('node_feat_dim', test_graphs[0].x.shape[1])
    hybrid_feat_dim = checkpoint.get('hybrid_feat_dim', test_graphs[0].hybrid.shape[0])
    hidden_dim = checkpoint.get('hidden_dim', hidden_dim)
    num_layers = checkpoint.get('num_layers', num_layers)
    dropout = checkpoint.get('dropout', dropout)
    
    model = HybridGCN(
        node_in_channels=node_feat_dim,
        hybrid_in_channels=hybrid_feat_dim,
        hidden_channels=hidden_dim,
        num_classes=len(CLASS_NAMES),
        num_layers=num_layers,
        dropout=dropout,
        embedding_dim=8
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Running Inference (SAVE PROBABILITIES!)
    print("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []  # NEW: Save full probability distributions
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            hybrid_batch = batch.hybrid.view(batch.num_graphs, -1)
            out = model(batch.x, batch.edge_index, batch.batch, hybrid_batch)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
            
    # 4. Metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Overall Accuracy
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {final_acc:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-Class Accuracy
    print("\nPer-Class Accuracy:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort classes by accuracy (ascending) to highlight issues
    sorted_indices = np.argsort(class_acc)
    for i in sorted_indices:
        print(f"{CLASS_NAMES[i]:<30}: {class_acc[i]:.4f}")
        
    # 5. Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics = {
        'accuracy': float(final_acc),
        'per_class_accuracy': {CLASS_NAMES[i]: float(acc) for i, acc in enumerate(class_acc)},
        'confusion_matrix': cm.tolist()
    }
    
    with open(OUTPUT_DIR / f"evaluation_{viewpoint}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {OUTPUT_DIR / f'evaluation_{viewpoint}.json'}")
    
    # 6. Generate Basic Confusion Matrix
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {viewpoint.upper()} View (Acc: {final_acc:.2%})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"confusion_matrix_{viewpoint}.png", dpi=150)
        print(f"Confusion matrix plot saved to {OUTPUT_DIR / f'confusion_matrix_{viewpoint}.png'}")
        plt.close()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")
    
    # 7. Generate Advanced Visualizations
    print("\n" + "="*60)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("="*60 + "\n")
    
    try:
        plot_threshold_sensitivity(all_probs, all_labels, viewpoint)
    except Exception as e:
        print(f"Error generating threshold sensitivity plot: {e}")
    
    try:
        plot_cross_threshold_heatmap(all_probs, all_labels, viewpoint)
    except Exception as e:
        print(f"Error generating cross-threshold heatmap: {e}")
    
    try:
        plot_stacked_distribution(all_probs, all_labels, viewpoint)
    except Exception as e:
        print(f"Error generating stacked distribution plot: {e}")
    
    try:
        plot_ground_truth_distribution(all_labels, viewpoint)
    except Exception as e:
        print(f"Error generating ground truth distribution plot: {e}")
    
    try:
        plot_per_class_accuracy(cm, viewpoint)
    except Exception as e:
        print(f"Error generating per-class accuracy plot: {e}")
    
    try:
        plot_classification_report_heatmap(all_labels, all_preds, viewpoint)
    except Exception as e:
        print(f"Error generating classification report heatmap: {e}")
    
    try:
        plot_training_history(viewpoint)
    except Exception as e:
        print(f"Error generating training history plot: {e}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, required=True,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to evaluate')
    args = parser.parse_args()
    
    evaluate_model(args.viewpoint)
