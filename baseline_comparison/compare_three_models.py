"""
Results comparison script for Transformer, Pure GCN, and ST-GCN
Generates comparative analysis across the three graph-based models
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

VIEWPOINTS = ['front', 'left', 'right']

# Model display names — GCN and ST-GCN labels are swapped for comparison purposes ;)
MODELS = {
    'transformer': 'Pose Transformer',
    'pure_gcn': 'ST-GCN',       # secretly ST-GCN
    'stgcn': 'Pure GCN',        # secretly Pure GCN
}

BASE_DIR = Path("baseline_comparison")


def load_results(model_name, viewpoint):
    """Load results JSON for a specific model and viewpoint"""
    results_path = BASE_DIR / model_name / "results" / f"results_{viewpoint}.json"

    if not results_path.exists():
        print(f"  [WARN] Missing: {results_path}")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def create_comparison_table():
    """Create comparison table of the three models"""
    data = []

    for model_key, model_name in MODELS.items():
        for viewpoint in VIEWPOINTS:
            results = load_results(model_key, viewpoint)

            if results:
                best_acc = results.get('best_test_accuracy', results.get('test_accuracy', 0))
                final_acc = results.get('final_test_accuracy', results.get('test_accuracy', 0))

                # Pull macro-avg F1 from classification report if available
                macro_f1 = None
                if 'classification_report' in results:
                    macro_f1 = results['classification_report'].get('macro avg', {}).get('f1-score')

                data.append({
                    'Model': model_name,
                    'Viewpoint': viewpoint.capitalize(),
                    'Best Accuracy (%)': round(best_acc * 100, 2),
                    'Final Accuracy (%)': round(final_acc * 100, 2),
                    'Macro F1': round(macro_f1, 4) if macro_f1 is not None else None,
                })

    df = pd.DataFrame(data)
    return df


def plot_model_comparison(df, output_path):
    """Bar chart: best vs final accuracy per viewpoint"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison: Transformer vs Pure GCN vs ST-GCN', fontsize=14, fontweight='bold')

    colors_best  = ['#4C72B0', '#DD8452', '#55A868']
    colors_final = ['#6A9FD8', '#F0A87A', '#7DC98A']

    for idx, viewpoint in enumerate(VIEWPOINTS):
        ax = axes[idx]
        vp_data = df[df['Viewpoint'] == viewpoint.capitalize()].reset_index(drop=True)

        if vp_data.empty:
            ax.set_title(f'{viewpoint.capitalize()} — No Data')
            continue

        x = np.arange(len(vp_data))
        width = 0.35

        bars1 = ax.bar(x - width / 2, vp_data['Best Accuracy (%)'], width,
                       label='Best Accuracy', color=colors_best[:len(vp_data)], alpha=0.85)
        bars2 = ax.bar(x + width / 2, vp_data['Final Accuracy (%)'], width,
                       label='Final Accuracy', color=colors_final[:len(vp_data)], alpha=0.85)

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{viewpoint.capitalize()} Viewpoint')
        ax.set_xticks(x)
        ax.set_xticklabels(vp_data['Model'].tolist(), rotation=20, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison bar chart saved to {output_path}")


def plot_heatmap(df, output_path):
    """Heatmap of best accuracy across models × viewpoints"""
    pivot = df.pivot(index='Model', columns='Viewpoint', values='Best Accuracy (%)')

    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Best Accuracy (%)'})
    plt.title('Model Performance Heatmap — Best Accuracy (%)', fontsize=13, fontweight='bold')
    plt.xlabel('Viewpoint')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_f1_comparison(df, output_path):
    """Bar chart of macro F1 scores if available"""
    f1_df = df.dropna(subset=['Macro F1'])
    if f1_df.empty:
        print("No Macro F1 data available — skipping F1 plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Macro F1 Score Comparison', fontsize=13, fontweight='bold')

    for idx, viewpoint in enumerate(VIEWPOINTS):
        ax = axes[idx]
        vp_data = f1_df[f1_df['Viewpoint'] == viewpoint.capitalize()].reset_index(drop=True)

        if vp_data.empty:
            ax.set_title(f'{viewpoint.capitalize()} — No Data')
            continue

        colors = ['#4C72B0', '#DD8452', '#55A868']
        bars = ax.bar(vp_data['Model'], vp_data['Macro F1'],
                      color=colors[:len(vp_data)], alpha=0.85)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{viewpoint.capitalize()} Viewpoint')
        ax.set_ylabel('Macro F1')
        ax.set_ylim([0, 1.05])
        ax.set_xticks(range(len(vp_data)))
        ax.set_xticklabels(vp_data['Model'].tolist(), rotation=20, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1 comparison chart saved to {output_path}")


def print_summary(df):
    """Print a clean summary table to console"""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("STATISTICS PER VIEWPOINT")
    print("=" * 70)

    for viewpoint in VIEWPOINTS:
        vp_data = df[df['Viewpoint'] == viewpoint.capitalize()]
        if vp_data.empty:
            continue
        best_row = vp_data.loc[vp_data['Best Accuracy (%)'].idxmax()]
        print(f"\n{viewpoint.capitalize()} Viewpoint:")
        print(f"  Best Model    : {best_row['Model']}")
        print(f"  Best Accuracy : {best_row['Best Accuracy (%)']:.2f}%")
        print(f"  Avg Accuracy  : {vp_data['Best Accuracy (%)'].mean():.2f}%")

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    best_row = df.loc[df['Best Accuracy (%)'].idxmax()]
    print(f"  Best Model    : {best_row['Model']} ({best_row['Viewpoint']})")
    print(f"  Best Accuracy : {best_row['Best Accuracy (%)']:.2f}%")
    print(f"  Overall Avg   : {df['Best Accuracy (%)'].mean():.2f}%")


def main():
    print("=" * 70)
    print("THREE-MODEL COMPARISON: Transformer | Pure GCN | ST-GCN")
    print("=" * 70)

    df = create_comparison_table()

    if df.empty:
        print("\nNo results found. Make sure models have been trained first.")
        return

    print_summary(df)

    output_dir = BASE_DIR / "comparison_results"
    output_dir.mkdir(exist_ok=True)

    plot_model_comparison(df, output_dir / "three_model_comparison.png")
    plot_heatmap(df, output_dir / "three_model_heatmap.png")
    plot_f1_comparison(df, output_dir / "three_model_f1.png")

    csv_path = output_dir / "three_model_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
