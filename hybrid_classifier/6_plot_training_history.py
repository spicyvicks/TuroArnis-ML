"""
Step 6: Plot Training History
Reads history.json generated during training and plots loss/accuracy curves.
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("hybrid_classifier/analysis")
MODELS_DIR = Path("hybrid_classifier/models")

def plot_history(viewpoint_filter=None):
    """Plot training history for a specific model"""
    
    suffix = f"_{viewpoint_filter}" if viewpoint_filter else ""
    history_file = MODELS_DIR / f"history{suffix}.json"
    
    if not history_file.exists():
        print(f"Error: History file not found at {history_file}")
        print("Did you train the model with the updated script?")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.set_title(f'Training Loss ({viewpoint_filter if viewpoint_filter else "All"})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'Accuracy ({viewpoint_filter if viewpoint_filter else "All"})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # Highlight best epoch
    best_epoch = np.argmax(history['test_acc']) + 1
    best_acc = max(history['test_acc'])
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.4f}')
    ax2.legend()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"training_curves{suffix}.png"
    plt.savefig(output_path)
    print(f"✓ Plots saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoint', type=str, default=None,
                        choices=['front', 'left', 'right'],
                        help='Viewpoint to plot (default: combined/all)')
    args = parser.parse_args()
    
    plot_history(args.viewpoint)
