# Investigation Report: March 2 Models vs Current Code

**Date:** 2026-04-22  
**Objective:** Identify differences between the 80%+ accuracy models (Feb 13, 2026) and current code causing performance degradation.

---

## Executive Summary

The **80% accuracy models** (commit 97070c0, Feb 13, 2026) were trained with significantly different hyperparameters and class configurations than the current code. **Four critical mismatches** have been identified that likely cause the performance drop:

1. **NODE_EMBED_DIM mismatch** (CRITICAL)
2. **Class count mismatch** - 12 vs 13 classes
3. **Class naming mismatch** - `neutral_stance` vs `neutral`
4. **Inconsistent hidden dimensions** across viewpoint models

---

## 1. Model File Analysis

### Deployment Package Models (Feb 11, 2026)

| Model | File | Hidden Dim | Node Embed Dim | Classes | fc.weight Shape |
|-------|------|------------|----------------|---------|-----------------|
| FRONT | `hybrid_gcn_v2_front.pth` | 128 | 8 | 12 | `[12, 128]` |
| LEFT | `hybrid_gcn_v2_left.pth` | 256 | 8 | 12 | `[12, 256]` |
| RIGHT | `hybrid_gcn_v2_right.pth` | 256 | 8 | 12 | `[12, 256]` |

**Key Finding:** LEFT and RIGHT models use `hidden_dim=256`, but FRONT uses `hidden_dim=128`. All use `NODE_EMBED_DIM=8`.

---

## 2. Architecture Comparison

### February 13, 2026 (80% Accuracy Commit 97070c0)

**File:** `hybrid_classifier/4c_train_hybrid_gcn_v2.py`

```python
# Hyperparameters
hidden_dim = 128          # Changed from 256 in commit 812b560
lr = 0.001                # Adam optimizer learning rate
embedding_dim = 8         # Node embedding dimension
dropout = 0.5
num_layers = 3
epochs = 150

# Architecture from training/train_gcn.py (imported CLASS_NAMES)
CLASS_NAMES = [
    'crown_thrust_correct',
    'left_chest_thrust_correct',
    'left_elbow_block_correct',
    'left_eye_thrust_correct',
    'left_knee_block_correct',
    'left_temple_block_correct',
    'neutral_stance',           # ← Named 'neutral_stance'
    'right_chest_thrust_correct',
    'right_elbow_block_correct',
    'right_eye_thrust_correct',
    'right_knee_block_correct',
    'right_temple_block_correct',
    'solar_plexus_thrust_correct'
]
# TOTAL: 13 classes
```

### Current Code (HEAD)

**File:** `hybrid_classifier/4c_train_hybrid_gcn_v2.py`

```python
# Hyperparameters (as of HEAD)
HIDDEN_DIM = 128              # ← Same
LEARNING_RATE = 0.005         # ← 5x higher! (was 0.001)
NODE_EMBED_DIM = 16           # ← DOUBLED from 8!
DROPOUT = 0.5                 # ← Same
NUM_LAYERS = 3                # ← Same
EPOCHS = 150
PATIENCE = 20

# Class names defined locally
CLASS_NAMES = [
    'front_left_chest_thrust', 'front_right_chest_thrust',
    'front_crown_thrust', 'left_crown_thrust', 'right_crown_thrust',
    'left_jab', 'right_jab',
    'front_left_downward_block', 'front_right_downward_block',
    'left_outward_block', 'right_outward_block', 'left_waist_block',
    'neutral'                    # ← Named 'neutral' (different!)
]
# TOTAL: 13 classes (but different names)
```

---

## 3. Critical Mismatches

### 🔴 MISMATCH #1: NODE_EMBED_DIM

| Version | Value | Impact |
|---------|-------|--------|
| Feb 13 (80% models) | **8** | Model was trained with 8-dim embeddings |
| Current code | **16** | Code expects 16-dim embeddings |

**Problem:** The deployment models have `node_embedding.weight` shape `[35, 8]`, but current code initializes `nn.Embedding(35, 16)`. This creates a **shape mismatch** when loading state dicts.

**Evidence from model inspection:**
```
FRONT model: node_embedding.weight: [35, 8]
Current code: NODE_EMBED_DIM = 16
```

---

### 🔴 MISMATCH #2: Class Count

| Model | Classes | Notes |
|-------|---------|-------|
| FRONT deployment | 12 | No neutral class |
| LEFT deployment | 12 | No neutral class |
| RIGHT deployment | 12 | No neutral class |
| Current code | 13 | Includes 'neutral' |

**Problem:** The deployment models were trained with 12 classes (no neutral), but the code expects 13 classes. The final classification layer (`fc.weight`) has shape `[12, 128]` or `[12, 256]` in models, but current code creates `fc2` with 13 outputs.

**History:**
- Commit `00a0005` (Apr 20): Removed neutral class to align with app
- Commit `7340e8d` (Apr 20): **Restored** neutral class because "models were trained with it"
- **BUT:** The deployment models (Feb 11) still have only 12 classes!

---

### 🔴 MISMATCH #3: Class Names

| Version | Neutral Class Name |
|---------|-------------------|
| Feb 13 training | `neutral_stance` |
| Current code | `neutral` |

**Problem:** When loading feature templates from `feature_templates.json`, the code looks for keys like `"{viewpoint}_{class_name}"`. If templates use `neutral_stance` but code looks for `neutral`, similarity features will return zeros for the neutral class.

**Evidence from feature_templates.json:**
```json
{
  "front_crown_thrust_correct": { ... },
  "front_neutral_stance": { ... },  // ← Old naming
  ...
}
```

---

### 🟡 MISMATCH #4: Hidden Dimension Inconsistency

| Model | Hidden Dim | Status |
|-------|------------|--------|
| FRONT | 128 | ✅ Matches current code |
| LEFT | 256 | ❌ Different from current code |
| RIGHT | 256 | ❌ Different from current code |

**Problem:** Current code uses `HIDDEN_DIM = 128` uniformly, but LEFT/RIGHT models were trained with 256. This might not affect inference (state dict loading), but it indicates inconsistent training configurations.

---

## 4. Feature Generation Changes

### February 13 (80% models)

- **Dataset:** `dataset_augmented/`
- **Output:** `hybrid_features_v2/`
- **Class names in feature generation:** From `training/train_gcn.py` - `neutral_stance`
- **Stick correction:** No Method 4 (pinky snap) correction
- **3D angles:** World landmarks available but not consistently used

### Current Code

- **Dataset:** `dataset_split/`
- **Output:** `hybrid_features_v3/`
- **Class names:** Defined locally with `neutral`
- **Stick correction:** Method 4 with pinky snap and foreshortening check
- **3D angles:** Always uses world landmarks for angles

**Key Changes (from git diff):**
```diff
-DATASET_ROOT = Path("dataset_augmented")
+DATASET_ROOT = Path("dataset_split")
 
-OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v2")
+OUTPUT_DIR = Path("hybrid_classifier/hybrid_features_v3")
```

---

## 5. Training Pipeline Changes

### Optimizations Added After Feb 13

Current code has these additional optimizations that may interfere with loading old models:

1. **WeightedRandomSampler** - For class imbalance
2. **Xavier initialization** - `model.apply(init_weights)` overwrites loaded weights
3. **track_running_stats=False** in BatchNorm - Different from Feb 13
4. **Learning rate 0.005** - 5x higher than Feb 13's 0.001
5. **ReduceLROnPlateau scheduler** - With different patience

---

## 6. Recommendations to Fix

### Immediate Fix (to use Feb 13 models)

1. **Revert NODE_EMBED_DIM to 8:**
   ```python
   NODE_EMBED_DIM = 8  # Was 16
   ```

2. **Use 12 classes (remove neutral) for old models:**
   ```python
   # For deployment models (trained without neutral)
   CLASS_NAMES = [
       'crown_thrust_correct', 'left_chest_thrust_correct', ...
       'solar_plexus_thrust_correct'
   ]  # 12 classes
   ```

3. **Or rename neutral class in templates:**
   ```python
   # If using 13 classes, ensure template keys match
   class_name = 'neutral_stance' if class_name == 'neutral' else class_name
   ```

4. **Update hidden_dim for LEFT/RIGHT:**
   ```python
   HIDDEN_DIM = 256 if viewpoint in ['left', 'right'] else 128
   ```

### Proper Fix (retrain with current config)

If retraining is preferred over fixing the mismatches:

1. Regenerate features with current `2b_generate_node_hybrid_features.py`
2. Train new models with consistent:
   - NODE_EMBED_DIM = 16
   - HIDDEN_DIM = 128 (or 256 for all)
   - 13 classes with `neutral` naming
   - Learning rate 0.005
3. Update deployment package with new models

---

## 7. Timeline of Changes

```
Feb 11, 2026: Deployment models created (80% accuracy)
  ↓
Feb 13, 2026: Commit 97070c0 - "80 achieved at front" (tuned to 128 hidden_dim)
  ↓
Apr 20, 2026: Commit 00a0005 - Removed neutral class (mistake)
  ↓
Apr 20, 2026: Commit 7340e8d - Restored neutral class
  ↓
Apr 22, 2026: Current HEAD - NODE_EMBED_DIM=16, new class names, v3 features
```

---

## 8. Root Cause Summary

The performance degradation is caused by **loading old models (trained with NODE_EMBED_DIM=8, 12 classes, old class names) into code that expects NODE_EMBED_DIM=16, 13 classes, new class names**.

**Primary culprit:** `NODE_EMBED_DIM` change from 8 → 16 means state dict loading fails or shapes mismatch silently.

**Secondary culprit:** Feature templates may use `neutral_stance` while code looks for `neutral`, causing neutral class to have all-zero hybrid features.

---

## Files Involved

- `hybrid_classifier/4c_train_hybrid_gcn_v2.py` - Training script
- `hybrid_classifier/2b_generate_node_hybrid_features.py` - Feature generation
- `deployment_package/models/hybrid_gcn_v2_*.pth` - Model weights
- `deployment_package/src/model_architecture.py` - Inference architecture
- `deployment_package/src/feature_extraction.py` - Inference features
- `deployment_package/src/feature_templates.json` - Hybrid feature statistics

---

**Next Steps:** Apply the "Immediate Fix" recommendations to load Feb 13 models correctly, or proceed with full retraining using consistent configurations.
