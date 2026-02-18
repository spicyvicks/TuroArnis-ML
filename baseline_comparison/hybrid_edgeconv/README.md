# Hybrid EdgeConv

## Concept
**Hybrid EdgeConv** = Dynamic graph learning + Expert features

Instead of using a fixed skeleton like Hybrid GCN, EdgeConv dynamically learns which joints should be connected based on feature similarity.

## Key Design Choices

Based on lessons from Hybrid ST-GCN experiments:

### 1. **Moderate Depth: 3 Layers**
- Learned from Hybrid ST-GCN: Deep networks (9 layers) overfit expert features
- 3 EdgeConv layers balances capacity and generalization

### 2. **Reduced k-NN: k=15**
- Original EdgeConv uses k=20
- Reduced to k=15 to prevent overfitting on small dataset (~2,500 samples)

### 3. **Moderate Dropout: 0.4**
- Lower than Hybrid GCN's 0.5
- EdgeConv's dynamic graphs provide implicit regularization

### 4. **Standard Regularization**
- Weight Decay: 5e-3
- Label Smoothing: 0.1
- Gradient Clipping: 1.0

## Expected Performance

| Model | Graph Type | Expected Acc |
|-------|-----------|--------------|
| Hybrid GCN | Fixed Skeleton | ~85% |
| **Hybrid EdgeConv** | **Dynamic k-NN** | **~75-82%** |
| Pure EdgeConv | Dynamic k-NN (raw) | ~50% |

## Training

```bash
python baseline_comparison/hybrid_edgeconv/train_hybrid_edgeconv.py --viewpoint front --k 15 --dropout 0.4
```

## Why It Might Work Better Than Hybrid ST-GCN

1. **Dynamic graphs adapt to expert features**: EdgeConv can learn that "elbow angle" is more similar to "shoulder angle" than to "knee angle"
2. **Moderate complexity**: 3 layers instead of 9
3. **Feature-based connections**: k-NN in feature space is more meaningful with expert features than with raw XYZ

## Why It Might Not Beat Hybrid GCN

1. **Fixed skeleton has domain knowledge**: The human skeleton structure is known and optimal
2. **Dynamic graphs add noise**: Learning connections might introduce instability
3. **Expert features already encode relationships**: Angles and distances already capture joint relationships

This experiment tests: **"Can learned graph structure beat anatomical knowledge?"**
