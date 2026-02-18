"""
Results comparison script for all baseline models
Generates comparative analysis across MLP, XGBoost, CapsNet, Transformer, EdgeConv, and ST-GCN
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

VIEWPOINTS = ['front', 'left', 'right']
MODELS = {
    'mlp': 'Pure MLP',
    'xgboost': 'XGBoost',
    'capsnet': 'CapsNet',
    'pure_gcn': 'Pure GCN',
    'transformer': 'Pose Transformer',
    'edgeconv': 'EdgeConv',
    'stgcn': 'ST-GCN'
}

BASE_DIR = Path("baseline_comparison")

def load_results(model_name, viewpoint):
    """Load results JSON for a specific model and viewpoint"""
    results_path = BASE_DIR / model_name / "results" / f"results_{viewpoint}.json"
    
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def create_comparison_table():
    """Create comparison table of all models"""
    data = []
    
    for model_key, model_name in MODELS.items():
        for viewpoint in VIEWPOINTS:
            results = load_results(model_key, viewpoint)
            
            if results:
                # Handle different JSON formats
                # XGBoost uses 'test_accuracy', others use 'best_test_accuracy'
                best_acc = results.get('best_test_accuracy', results.get('test_accuracy', 0))
                final_acc = results.get('final_test_accuracy', results.get('test_accuracy', 0))
                
                data.append({
                    'Model': model_name,
                    'Viewpoint': viewpoint.capitalize(),
                    'Best Accuracy': best_acc * 100,
                    'Final Accuracy': final_acc * 100
                })
    
    df = pd.DataFrame(data)
    return df

def plot_model_comparison(df, output_path):
    """Plot comparison of all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, viewpoint in enumerate(VIEWPOINTS):
        ax = axes[idx]
        viewpoint_data = df[df['Viewpoint'] == viewpoint.capitalize()]
        
        if not viewpoint_data.empty:
            x = np.arange(len(viewpoint_data))
            width = 0.35
            
            ax.bar(x - width/2, viewpoint_data['Best Accuracy'], width, 
                   label='Best Accuracy', alpha=0.8)
            ax.bar(x + width/2, viewpoint_data['Final Accuracy'], width,
                   label='Final Accuracy', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{viewpoint.capitalize()} Viewpoint')
            ax.set_xticks(x)
            ax.set_xticklabels(viewpoint_data['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")

def plot_heatmap(df, output_path):
    """Plot heatmap of model performance across viewpoints"""
    pivot_data = df.pivot(index='Model', columns='Viewpoint', values='Best Accuracy')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Model Performance Heatmap (Best Accuracy %)')
    plt.xlabel('Viewpoint')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")

def generate_latex_table(df, output_path):
    """Generate LaTeX table for paper"""
    # Calculate average across viewpoints
    avg_data = df.groupby('Model')['Best Accuracy'].mean().reset_index()
    avg_data.columns = ['Model', 'Average Accuracy']
    
    # Merge with original data
    pivot_best = df.pivot(index='Model', columns='Viewpoint', values='Best Accuracy')
    pivot_best['Average'] = avg_data['Average Accuracy']
    
    # Generate LaTeX
    latex = pivot_best.to_latex(float_format='%.2f', caption='Baseline Model Comparison',
                                 label='tab:baseline_comparison')
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to {output_path}")

def main():
    print("="*80)
    print("BASELINE MODELS COMPARISON")
    print("="*80)
    
    # Create comparison table
    df = create_comparison_table()
    
    if df.empty:
        print("No results found. Train models first.")
        return
    
    # Display table
    print("\nComparison Table:")
    print(df.to_string(index=False))
    
    # Calculate statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    for viewpoint in VIEWPOINTS:
        viewpoint_data = df[df['Viewpoint'] == viewpoint.capitalize()]
        if not viewpoint_data.empty:
            print(f"\n{viewpoint.capitalize()} Viewpoint:")
            print(f"  Best Model: {viewpoint_data.loc[viewpoint_data['Best Accuracy'].idxmax(), 'Model']}")
            print(f"  Best Accuracy: {viewpoint_data['Best Accuracy'].max():.2f}%")
            print(f"  Average Accuracy: {viewpoint_data['Best Accuracy'].mean():.2f}%")
    
    # Overall statistics
    print(f"\nOverall:")
    print(f"  Best Model: {df.loc[df['Best Accuracy'].idxmax(), 'Model']} "
          f"({df.loc[df['Best Accuracy'].idxmax(), 'Viewpoint']})")
    print(f"  Best Accuracy: {df['Best Accuracy'].max():.2f}%")
    print(f"  Average Accuracy: {df['Best Accuracy'].mean():.2f}%")
    
    # Generate visualizations
    output_dir = BASE_DIR / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    plot_model_comparison(df, output_dir / "model_comparison.png")
    plot_heatmap(df, output_dir / "performance_heatmap.png")
    generate_latex_table(df, output_dir / "comparison_table.tex")
    
    # Save CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved to {csv_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
