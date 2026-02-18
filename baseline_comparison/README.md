# Baseline Model Comparison for Arnis Pose Classification

This directory contains three baseline models to compare against the Hybrid GCN.

**IMPORTANT**: All models train on **viewpoint-specific** data. You must specify `--viewpoint front`, `--viewpoint left`, or `--viewpoint right` when training.

## Models

### 1. XGBoost (Traditional ML Baseline)
**Question**: Do we even need Deep Learning?

- **Input**: Flattened 210-dim vector (35 nodes × 6 features)
- **Method**: Gradient-boosted decision trees
- **Strengths**: Fast training, interpretable feature importance
- **Expected Accuracy**: ~73-76%

**Train**:
```bash
python baseline_comparison/xgboost/train_xgboost.py --viewpoint front
python baseline_comparison/xgboost/train_xgboost.py --viewpoint left
python baseline_comparison/xgboost/train_xgboost.py --viewpoint right
```

**Grid Search** (optional):
```bash
python baseline_comparison/xgboost/train_xgboost.py --grid_search
```

### 2. Pure MLP (Deep Learning Baseline)
**Question**: Does the graph structure actually matter?

- **Input**: Flattened 210-dim vector
- **Architecture**: 4-layer MLP (512→256→128→12) with Dropout & BatchNorm
- **Strengths**: Tests if explicit anatomical connections help
- **Expected Accuracy**: ~77-80%

**Train**:
```bash
python baseline_comparison/mlp/train_mlp.py --viewpoint front --epochs 150
python baseline_comparison/mlp/train_mlp.py --viewpoint left --epochs 150
python baseline_comparison/mlp/train_mlp.py --viewpoint right --epochs 150
```

### 3. CapsNet (Advanced Alternative)
**Question**: Is routing-by-agreement better for viewpoint invariance?

- **Input**: 7 anatomical groups (35 nodes grouped into regions)
- **Architecture**: Primary capsules → Dynamic routing → Digit capsules
- **Strengths**: Explicitly models part-whole hierarchies
- **Expected Accuracy**: ~74-78%

**Train**:
```bash
python baseline_comparison/capsnet/train_capsnet.py --viewpoint front --epochs 200
python baseline_comparison/capsnet/train_capsnet.py --viewpoint left --epochs 200
python baseline_comparison/capsnet/train_capsnet.py --viewpoint right --epochs 200
```

## Output Files

Each model generates:
- **Model checkpoint**: `results/<model>_best_<viewpoint>.pth` (or `.json` for XGBoost)
- **Results JSON**: `results/results_<viewpoint>.json` (accuracy, classification report)
- **Confusion Matrix**: `results/confusion_matrix_<viewpoint>.png`
- **Training History**: `results/training_history_<viewpoint>.png` (MLP & CapsNet only)
- **Feature Importance**: `results/feature_importance_<viewpoint>.png` (XGBoost only)

## Environment Setup

All models use the existing `venv_gcn` environment. Ensure you have:
```bash
pip install xgboost pandas tqdm
```

## Data Source

All models load data from:
```
hybrid_classifier/hybrid_features_v4/
├── train_features_front.pt
├── test_features_front.pt
├── train_features_left.pt
├── test_features_left.pt
├── train_features_right.pt
└── test_features_right.pt
```

Each `.pt` file contains:
- `node_features`: (N, 35, 6) tensor
- `labels`: (N,) tensor
- `viewpoints`: list of viewpoint strings

## Comparison Metrics

After training all models, compare:
1. **Test Accuracy** (primary metric)
2. **Per-class F1 scores** (from classification reports)
3. **Confusion patterns** (which techniques are confused)
4. **Training stability** (variance across viewpoints)
5. **Computational cost** (training time, model size)

## Advanced Baselines

See [ADVANCED_BASELINES.md](ADVANCED_BASELINES.md) for detailed documentation on:
- **Pose Transformer**: Self-attention mechanism (80-84% expected)
- **EdgeConv**: Dynamic k-NN graph learning (81-85% expected)
- **ST-GCN**: Spatio-temporal graph convolution (79% expected for T=1)

**Quick Start**:
```bash
# Train all advanced baselines
python baseline_comparison/train_all_advanced.py --models all --viewpoints all

# Compare all results
python baseline_comparison/compare_results.py
```

## Expected Results Summary

| Model | Expected Accuracy | Key Limitation |
|-------|------------------|----------------|
| XGBoost | 73-76% | No spatial structure |
| Pure MLP | 77-80% | No anatomical constraints |
| CapsNet | 74-78% | Training instability |
| **Pure GCN** | **75-78%** | No expert features |
| **Transformer** | **80-84%** | O(N²) complexity, no anatomy |
| **EdgeConv** | **81-85%** | Unstable under occlusion |
| **ST-GCN** | **79%** | No expert features (T=1) |
| **Hybrid GCN** | **~85%** | *(Your baseline to beat)* |
