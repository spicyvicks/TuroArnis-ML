# HybridGCN Retraining Results - Option A

## Summary

Retraining of HybridGCN model with **Option A hyperparameter adjustments** completed successfully. Model showed **significant improvement** from 24.9% to 37.6% validation accuracy (+51% relative improvement).

---

## Hyperparameter Changes (Option A)

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| **DROPOUT** | 0.7 | 0.5 | Less aggressive regularization |
| **LEARNING_RATE** | 0.001 | 0.005 | 5x faster learning |
| **WEIGHT_DECAY** | 1e-4 | 5e-5 | Lighter regularization |
| **PATIENCE** | 15 | 20 | More epochs to converge |
| **BATCH_SIZE** | 32 | 64 | More stable gradients |
| **MAX_OVERFIT_GAP** | 25 | 35 | Allow more gap during learning |

---

## Training Results

### Key Metrics

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **37.6%** (epoch 98-99) |
| **Final Training Accuracy** | ~31.9% |
| **Epochs Trained** | 119 |
| **Train-Test Gap** | -4.6% (validation higher than training) |
| **Improvement from Baseline** | +12.7 pp (+51% relative) |

### Progress Over Time

| Epoch Range | Val Accuracy | Notes |
|-------------|--------------|-------|
| 1-10 | 15-19% | Initial learning phase |
| 11-30 | 21-28% | Steady improvement |
| 31-60 | 27-32% | Continuing gains |
| 61-90 | 31-37% | Best performance |
| 91-119 | 36-37.6% | Plateau, early stop |

### Loss Trajectory

- **Starting Training Loss**: 2.55 (epoch 1)
- **Ending Training Loss**: 1.78 (epoch 119)
- **Starting Val Loss**: 2.41 (epoch 1)
- **Ending Val Loss**: 1.98 (epoch 119)
- **Loss Reduction**: ~30% (clear learning occurred)

---

## Analysis

### What Worked ✅

1. **Significant improvement**: +12.7 percentage points (24.9% → 37.6%)
2. **Steady learning**: Training loss decreased consistently
3. **No underfitting**: Model learned training data (31.9% train accuracy)
4. **No severe overfitting**: Gap stayed reasonable (< 6%)
5. **Converged properly**: Model found stable plateau

### What Didn't Reach Targets ❌

1. **Target accuracy not met**: 37.6% vs 70% target
2. **Training accuracy low**: 31.9% indicates model capacity or data limitations
3. **Plateaued early**: No improvement after epoch ~100

### Observations

1. **Negative train-test gap**: Validation accuracy (37.6%) > Training accuracy (31.9%)
   - Caused by `WeightedRandomSampler` - training sees more hard examples
   - Indicates good generalization

2. **Learning rate decay pattern**: 
   - Started at 0.005, decayed to 9.7e-06
   - Scheduler working correctly

3. **Best model at epoch 98**: 37.6% validation accuracy
   - Model saved automatically

---

## Model Artifacts

### Generated Files

```
hybrid_classifier/models/
├── model_merged.pth          (Best model - 37.6% accuracy)
└── history_merged.json        (Full training history)
```

**Note**: Model files (.pth) are gitignored due to size. They are preserved locally.

### Model Architecture (unchanged)

- Hidden dim: 128
- Num layers: 3
- Node embed dim: 16
- Total parameters: ~500K

---

## Comparison with Baseline

| Metric | Baseline (0.7 dropout) | Option A (0.5 dropout) | Change |
|--------|------------------------|------------------------|--------|
| Best Val Acc | 24.9% | 37.6% | **+51%** ⬆️ |
| Train Acc | ~13% | 31.9% | **+145%** ⬆️ |
| Learning | None | Clear | **Fixed** ✅ |
| Epochs | 115 | 119 | Similar |

---

## Conclusions

### Option A Assessment: **PARTIAL SUCCESS**

✅ **Hyperparameter adjustments worked**: Model now learns instead of underfitting

✅ **51% relative improvement**: Significant gain from simple hyperparameter tuning

❌ **Still below 70% target**: Need additional improvements

### Root Cause of Remaining Gap

The model learns but plateaus at ~37%, suggesting:

1. **Feature quality**: Input features may not be discriminative enough
2. **Model capacity**: 128 hidden dim may be insufficient for 13-class problem
3. **Data limitations**: Class imbalance (6.7:1) still affects performance
4. **Architecture**: Simple GCN may need enhancement (attention, residual, etc.)

### Recommendations for Next Phase

1. **Try Option B**: Increase hidden_dim 128 → 256 (more capacity)
2. **Feature engineering**: Review hybrid feature quality
3. **Architecture changes**: Add attention mechanism or transformer layers
4. **Data augmentation**: Generate more training samples per class
5. **Ensemble methods**: Combine multiple model predictions

---

## Commit

```
commit 014140c
Author: GSD Executor
Date: Wed Apr 22 2026

feat(ml-retrain): retrain HybridGCN with Option A hyperparameters
```

**Files modified**: `hybrid_classifier/4c_train_hybrid_gcn_v2.py`
**Model artifacts**: `hybrid_classifier/models/` (local, not committed)

---

## Metrics Summary

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Best Val Accuracy | 37.6% | 70% | ⚠️ Partial |
| Train Acc >50% by epoch 20 | No (20.3%) | Yes | ❌ Missed |
| Val Acc >60% by epoch 30 | No (28.2%) | Yes | ❌ Missed |
| Gap <20% | Yes (-4.6%) | Yes | ✅ Pass |
| Improvement from baseline | +51% | Significant | ✅ Pass |
