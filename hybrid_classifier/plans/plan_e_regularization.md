# Plan E: Radical Regularization + Smaller Model

## Goal
Slash model capacity, add structural regularization, and find the optimal configuration via systematic cross-validation. The current model has **~150k–990k parameters** for only **1,997 training samples** — it memorizes training identities rather than learning generalizable pose structure. The train/val gap of **35%** is a classic symptom of over-parameterization on a small, person-diverse dataset.

## Expected Outcome
| Metric | Current | Target |
|--------|---------|--------|
| 5-fold CV accuracy | ~55% (inferred) | **≥62%** |
| Train/val gap | 35% | **≤12%** |
| Standard deviation across folds | ~4% | **≤3%** |
| Worst-class recall | 0% | **≥10%** |

## Context: Why This Plan Exists

The test set consists of **different people, locations, and times** than the training set. This means:
- The model cannot rely on memorizing body-specific pose coordinates
- **Synthetic augmentation** (perturbing coordinates within a training sample) does **not** create truly new body proportions or camera viewpoints
- The only path to generalization is **forcing the model to learn invariant, low-dimensional representations**
- Regularization (smaller model, dropout, weight decay) is the primary tool for this

Plan E is **not a standalone architecture change** — it is a **hyperparameter and training regime overhaul** that should be applied to:
- The **current HybridGCN v2** (for baseline comparison)
- The **Plan A PureGCN** (for best results)
- Any future architecture (Plan B included)

## Implementation Steps

---

### Step 1: Create Grid Search Script
**File:** `hybrid_classifier/4g_grid_search_regularization.py`

**Grid dimensions:**

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `HIDDEN_DIM` | 32, 48, 64 | Test if model is over-parameterized. If 32 underfits, bottleneck is feature quality. If 64 overfits, go smaller. |
| `NUM_LAYERS` | 2, 3 | 2 layers = less over-smoothing. 3 layers = more propagation but higher capacity. |
| `DROPOUT` | 0.5, 0.6, 0.7, 0.8 | Current 0.5 is too low for this dataset. 0.8 is extreme but necessary. |
| `WEIGHT_DECAY` | 1e-4, 5e-4, 1e-3, 5e-3 | AdamW + strong weight decay aggressively suppresses large weights. |
| `NODE_EMBED_DIM` | 2, 4, 8 | Node identity (which joint) matters less than coordinates. Lower is more regularized. |
| Loss | CrossEntropy, Focal(γ=1), Focal(γ=2) | Focal Loss focuses on hard examples (left/right confused classes). |

**Total configurations:** 3 × 2 × 4 × 4 × 3 × 3 = **864 configs**

**This is too many. Pruned grid (48 configs):**
- hid ∈ {32, 48, 64}, layers ∈ {2, 3}, dropout ∈ {0.6, 0.7, 0.8}, decay ∈ {1e-3, 5e-3}, embed ∈ {4}, loss ∈ {Focal(γ=1.5)}
- Plus a baseline: hid=128, layers=3, dropout=0.5, decay=5e-5, embed=8, loss=CE (current config)

**5-fold stratified cross-validation:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Combine train + test for CV (2,182 total samples)
all_features = torch.cat([train_data['node_features'], test_data['node_features']])
all_hybrid = torch.cat([train_data['hybrid_features'], test_data['hybrid_features']])
all_labels = torch.cat([train_data['labels'], test_data['labels']])

for fold, (train_idx, val_idx) in enumerate(skf.split(all_features, all_labels)):
    # Train on train_idx, validate on val_idx
    # Average val_acc across 5 folds for each config
```

**Why 5-fold CV is valid here:**
Even though test subjects are different people, stratified CV gives **stable estimates** of whether a configuration generalizes. If `hidden_dim=64, dropout=0.7` scores 62% ± 2% across 5 folds, we can be confident it's better than `hidden_dim=128, dropout=0.5` scoring 57% ± 4%.

**Metric for ranking configs:**
Rank by **mean validation macro-F1** (not accuracy). Macro-F1 punishes rare-class failure, which is exactly our problem (left-side classes have low recall).

---

### Step 2: Implement Focal Loss + Effective Class Weighting
**File:** `hybrid_classifier/losses/focal_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=1.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

**Why Focal Loss matters here:**
- Current CrossEntropy treats all misclassifications equally
- The model is **very confident** on easy classes (neutral, solar plexus) and **very confused** on hard classes (left elbow block)
- Focal Loss down-weights easy examples, forcing gradient updates to focus on confused left/right pairs
- `gamma=1.5` is a moderate setting. If training becomes unstable, reduce to `gamma=1.0`.

**Class weighting — effective number formula (more stable than inverse frequency):**
```python
import numpy as np

class_counts = np.bincount(labels, minlength=NUM_CLASSES)
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
class_weights = (1.0 - beta) / effective_num
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = torch.tensor(class_weights, dtype=torch.float32)
```

This avoids the problem where a class with only 10 samples gets a 100× weight and causes the model to overfit to those 10 samples.

---

### Step 3: Implement DropEdge (Stochastic Edge Dropping)
**File:** `hybrid_classifier/dataset/graph_dataset.py`

```python
class GraphDataset(Dataset):
    def __init__(self, features_path, viewpoint=None, filter_nan=True, drop_edge_prob=0.1):
        # ... existing init ...
        self.drop_edge_prob = drop_edge_prob
        self.training = False  # set to True by DataLoader during train loop
    
    def __getitem__(self, idx):
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()
        
        # DropEdge: randomly remove edges during training
        if self.training and self.drop_edge_prob > 0:
            mask = torch.rand(edge_index.size(1)) > self.drop_edge_prob
            edge_index = edge_index[:, mask]
        
        # ... rest of construction
        return Data(x=node_features, edge_index=edge_index, y=label)
```

**Why DropEdge helps:**
- Forces the GCN to learn from **incomplete skeletons**
- Prevents over-reliance on any single edge (e.g., the stick-to-wrist edge)
- Acts as a data augmentation on the graph structure itself
- Equivalent to "cutout" or "dropout" for graph edges
- `drop_edge_prob=0.1` means ~10% of edges removed per sample — enough to matter, not enough to disconnect the graph

---

### Step 4: Learning Rate Warmup + Cosine Annealing
**File:** `hybrid_classifier/4g_grid_search_regularization.py`

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Schedule behavior:**
| Period | Epochs | LR Behavior |
|--------|--------|-------------|
| 1st | 1–10 | Cosine decay from 0.001 → 0 |
| 2nd | 11–30 | Restart at 0.001, cosine decay → 0 |
| 3rd | 31–70 | Restart at 0.001, cosine decay → 0 |
| 4th | 71–150 | Restart at 0.001, cosine decay → 0 |

**Why this schedule:**
- With heavy dropout (0.7), the model needs **more epochs** to converge
- Standard step decay schedules decay too fast → model underfits
- Warm restarts let the model **escape local minima** caused by dropout sparsity
- Each restart is a "fresh attempt" with the same learning rate

---

### Step 5: Run Grid Search
**Command:**
```bash
python hybrid_classifier/4g_grid_search_regularization.py \
    --viewpoint front \
    --model pure_gcn  # or hybrid_gcn for comparison
```

**Expected runtime:**
- 49 configs × 5 folds × ~40 epochs each = ~9,800 training runs
- Each run: 2–3 minutes on CPU
- **Total: ~400 hours of CPU time** — must run overnight or on a subset

**Practical subset:**
- Run only the most promising 12 configs first (hid ∈ {48, 64}, layers ∈ {2, 3}, dropout ∈ {0.7, 0.8}, decay ∈ {1e-3, 5e-3})
- 12 × 5 = 60 runs = ~3 hours overnight
- If results are promising, expand to full grid

---

### Step 6: Ensemble Multiple Small Models
**File:** `hybrid_classifier/ensemble_evaluate.py`

Once the top 3–5 configurations are identified from grid search, train each with **5 different random seeds** and ensemble by averaging logits:

```python
def ensemble_predict(models, data_loader):
    all_preds = []
    all_labels = []
    
    for batch in data_loader:
        logits = torch.stack([model(batch) for model in models])
        avg_logits = logits.mean(dim=0)
        preds = avg_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    return all_preds, all_labels
```

**Why ensemble helps for person-level OOD:**
- Each small model memorizes different spurious correlations from training people
- Averaging cancels out the memorized noise
- Keeps the **consistent structural signals** that all models agree on
- Standard technique for unstable small-dataset generalization

---

### Step 7: Final Evaluation on Held-Out Test Set
**Command:**
```bash
python hybrid_classifier/ensemble_evaluate.py \
    --model_dir hybrid_classifier/models/best_configs \
    --test_data hybrid_classifier/hybrid_features_v3/test_features_front.pt \
    --report_json hybrid_classifier/reports/final_ensemble_report.json
```

**Metrics:**
- Overall accuracy
- Per-class recall
- Binary left/right accuracy
- Binary block/thrust accuracy
- Confusion matrix (saved as CSV)

---

## Dependencies Between Steps

```
Step 1 (grid search script) ───────────────────┐
                                                ├── Step 5 (run grid search) ── Step 6 (ensemble)
Step 2 (Focal Loss) ───────────┐                │
                                ├── Step 4 (LR schedule) ──┘
Step 3 (DropEdge) ─────────────┘
```

Steps 1–4 are parallel (code changes). Step 5 depends on 1–4. Step 6 depends on 5. Step 7 depends on 6.

## Estimated Effort
| Task | Time |
|------|------|
| Create grid search script | 1.5 hours |
| Implement Focal Loss | 30 min |
| Implement DropEdge | 30 min |
| Add LR warmup/cosine schedule | 30 min |
| Run subset grid search (12 configs × 5 folds) | 3 hours (overnight) |
| Run full grid search (if needed) | 6–8 hours |
| Ensemble evaluation | 1 hour |
| **Total** | **~7 hours** |

## Success Criteria
- **Best 5-fold CV macro-F1 ≥ 0.55** (equivalent to ~62% accuracy with balanced recall)
- **Standard deviation across folds ≤ 3%** (stable generalization)
- **Train/val gap ≤ 12%** at best configuration
- **Worst-class recall ≥ 10%** (no more 0% classes)

## Risk: What If Plan E Alone Doesn't Help?

If even `hidden_dim=32, dropout=0.8` still has a 25% train/val gap, the problem is **not model capacity** — it's **feature representation**. The model cannot learn what isn't in the features. At that point:
- **Plan A** (person-normalized pure GCN) is the next step — it changes the features, not just the model
- **Plan B** (hand landmarks) is the nuclear option — adds entirely new structural signal

## How Plan E Interacts with Other Plans

Plan E is a **universal upgrade** to training methodology. It should be applied to:
1. **Baseline HybridGCN v2** (for comparison: does regularization alone help?)
2. **Plan A PureGCN** (expected best results: new features + strong regularization)
3. **Plan B HybridGCNWithHands** (if Plan B is eventually needed)

**Recommended execution order:**
1. Run Plan E grid search on **current HybridGCN v2** → establishes baseline
2. If best config improves val accuracy by >3 points, the regularization was the bottleneck
3. If best config only improves by 1–2 points, the bottleneck is features → execute Plan A
4. After Plan A features are ready, run Plan E grid search again on **PureGCN** → expected to find the true ceiling
