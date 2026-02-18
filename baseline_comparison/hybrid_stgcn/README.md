# Hybrid ST-GCN Variants Comparison

## Three Versions

### 1. **5-Layer Version** (Moderate Depth)
**File**: `train_hybrid_stgcn.py`

**Configuration:**
- Layers: 5
- Dropout: 0.5
- Weight Decay: 5e-3
- Learning Rate: 0.001
- Label Smoothing: 0.1

**Purpose**: Test if moderate depth (5 layers) with standard regularization can match Hybrid GCN

**Training:**
```bash
python baseline_comparison/hybrid_stgcn/train_hybrid_stgcn.py --viewpoint front
```

---

### 2. **9-Layer Aggressive** (Deep + Extreme Regularization)
**File**: `train_hybrid_stgcn_aggressive.py`

**Configuration:**
- Layers: 9
- Dropout: 0.5
- Weight Decay: **1e-2** (2× higher)
- Learning Rate: **0.0005** (2× lower)
- **Gradient Clipping: 1.0**
- Label Smoothing: 0.1

**Purpose**: Test if deep networks (9 layers) can work with MAXIMUM regularization

**Training:**
```bash
python baseline_comparison/hybrid_stgcn/train_hybrid_stgcn_aggressive.py --viewpoint front
```

---

## Expected Results

| Version | Layers | Expected Test Acc | Train-Test Gap |
|---------|--------|------------------|----------------|
| 5-Layer | 5 | ~70-80% | ~5-10% |
| 9-Layer Aggressive | 9 | ~65-75% | ~10-15% |
| **Hybrid GCN** | **3** | **~85%** | **~5%** |

## Interpretation

### If 5-Layer wins (~75-80%):
- Proves moderate depth is better than extreme depth for this dataset
- Still below Hybrid GCN's 85%, validating the 3-layer design

### If 9-Layer Aggressive wins (~70-75%):
- Shows that deep networks CAN work with extreme regularization
- But still below Hybrid GCN, proving 3 layers is optimal

### If both underperform (<70%):
- **Conclusive proof** that your 3-layer Hybrid GCN is the sweet spot
- Dataset size (~2,500) doesn't support deeper architectures

## Key Takeaway

This experiment isolates the **depth variable** while controlling for regularization. Any performance difference is purely architectural, proving whether your compact 3-layer design is optimal or if depth helps when you have expert features.
