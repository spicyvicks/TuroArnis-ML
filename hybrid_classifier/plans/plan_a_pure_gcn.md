# Plan A: Pure GCN — Remove Hybrid MLP, Person-Normalize Nodes, Attention Pooling

## Goal
Replace the template-matching hybrid branch with a **person-invariant graph representation** that classifies from body geometry alone. The current hybrid MLP computes similarity to templates derived from the **training subjects' body proportions**. When test subjects have different limb lengths, standing distances, or camera heights, the "distance from template" features become meaningless noise.

## Expected Outcome
| Metric | Current | Target |
|--------|---------|--------|
| Val accuracy | 57.6% | **60–63%** |
| Left/right recall | 0–7% | **≥40%** |
| Train/val gap | 35% | **≤15%** |

## Context: Why This Plan Exists

The test set consists of **different people, locations, and times** than the training set. This means:
- Body proportions differ (arm length, torso height)
- Camera distances differ (features in absolute normalized coordinates shift)
- The current **template-matching hybrid features** are computed against training-subject templates → they actively mislead on test subjects
- The GCN branch memorizes training poses while the hybrid branch adds noise → severe overfitting

## Implementation Steps

---

### Step 1: Create `PureGCN` Model
**File:** `hybrid_classifier/models/pure_gcn.py`

**Architecture:**
```python
class PureGCN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64, num_layers=2):
        super().__init__()
        # Node identity embedding: 35 nodes -> 4 dims (reduced from 8)
        self.node_embedding = nn.Embedding(35, 4)
        
        # GCN layers: 2 layers instead of 3
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + 4, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention-based global pooling (learnable)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Single FC classifier (reduced capacity)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)
```

**Rationale:**
| Decision | Why |
|----------|-----|
| `hidden_dim=64` | 1,997 training samples; 64-dim is enough for 13 classes without memorizing individual identities |
| 2 GCN layers | 3 layers cause over-smoothing on small graphs; 2 layers preserve local node distinctiveness |
| Attention pooling | Replaces `global_mean_pool`. Model learns to focus on right hand, stick, and head while ignoring legs |
| No hybrid MLP | Removes the template-matching prior that fails on new body proportions |
| Dropout=0.7 | Aggressive regularization to prevent memorizing training identities |

---

### Step 2: Person-Normalize Node Features
**File:** `hybrid_classifier/2b_generate_node_hybrid_features.py`

**Current feature (person-dependent):**
```python
# [x, y, z, visibility, dist_to_hip_3d, angle_from_hip]
```
`dist_to_hip` depends on how far the person is from the camera and how tall they are.

**New feature (person-invariant):**
```python
def extract_node_features_normalized(pose_keypoints, stick_keypoints):
    """
    Returns [N, 8] per-node features:
    [x_norm, y_norm, z_norm, vis,
     rel_x_to_hip, rel_y_to_hip, rel_z_to_hip,
     limb_length_norm]
    """
    # 1. Compute torso scale: shoulder_width = dist(11, 12)
    shoulder_width = np.linalg.norm(pose_keypoints[11, :3] - pose_keypoints[12, :3])
    # 2. Compute hip center
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2
    
    features = []
    for i, kpt in enumerate(all_keypoints):
        x, y, z, vis = kpt
        # Normalize by torso scale (person-invariant)
        rel = (np.array([x, y, z]) - hip_center) / (shoulder_width + 1e-8)
        features.append([x, y, z, vis, rel[0], rel[1], rel[2], shoulder_width])
    
    return np.array(features, dtype=np.float32)
```

**Why this works:**
- Shoulder width is a stable proxy for "person scale" within a single frame
- Dividing all coordinates by shoulder width makes features **invariant to camera distance and body height**
- A `left_chest_thrust` from a 5'2" person and a 6'0" person will now have nearly identical normalized right-wrist positions

---

### Step 3: Dynamic Stick-to-Hand Edges
**File:** `hybrid_classifier/2b_generate_node_hybrid_features.py`

**Current edges (static, wrong):**
```python
SKELETON_EDGES = [(15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)]
```
This connects stick grip (33) to **both** left and right wrists. The stick is always in the **right hand**.

**New edges (dynamic, per-sample):**
```python
# After extracting stick and pose
if stick_keypoints is not None:
    grip = stick_keypoints[0, :2]  # x, y of grip
    dist_to_r_wrist = np.linalg.norm(grip - pose_keypoints[16, :2])
    dist_to_l_wrist = np.linalg.norm(grip - pose_keypoints[15, :2])
    
    if dist_to_r_wrist < dist_to_l_wrist:
        stick_edges = [(16, 33), (33, 16), (33, 34), (34, 33)]
    else:
        # YOLO may have flipped grip/tip; trust the closer hand
        stick_edges = [(15, 33), (33, 15), (33, 34), (34, 33)]
```

**Why this helps:**
- The graph structure itself encodes handedness
- GCN message passing flows from right wrist -> stick -> stick tip, making right-handedness a structural prior

---

### Step 4: Create Training Script `4f_train_pure_gcn.py`
**File:** `hybrid_classifier/4f_train_pure_gcn.py`

**Key hyperparameter changes from `4e_train_hybrid_gcn_v2_with_synthetic.py`:**

| Hyperparameter | Current (4e) | Plan A Target |
|----------------|-------------|---------------|
| `HIDDEN_DIM` | 128 | **64** |
| `NUM_LAYERS` | 3 | **2** |
| `NODE_EMBED_DIM` | 8 | **4** |
| `DROPOUT` | 0.5 | **0.7** |
| `WEIGHT_DECAY` | 5e-5 | **1e-3** |
| `BATCH_SIZE` | 64 | **32** |
| `MAX_OVERFIT_GAP` | 35% | **15%** |
| Loss | CrossEntropy | **FocalLoss(gamma=1.5)** |

**Focal Loss implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=1.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()
```

**Class weights:** Use effective number weighting (not inverse frequency) to avoid overfitting to tiny classes:
```python
beta = 0.9999
effective_num = 1.0 - np.power(beta, class_counts)
class_weights = (1.0 - beta) / effective_num
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
```

---

### Step 5: Regenerate `.pt` Files
**Command:**
```bash
python hybrid_classifier/2b_generate_node_hybrid_features.py --viewpoint front
```

This must be run because node features now use person-normalized coordinates.

---

### Step 6: Train + Evaluate
**Command:**
```bash
python hybrid_classifier/4f_train_pure_gcn.py --viewpoint front --epochs 150
```

**Metrics to report:**
- Overall validation accuracy
- Per-class recall (especially left-side classes: 1, 2, 3, 4, 5)
- Binary "left vs. right" accuracy (classes 1–5 vs. 6–10)
- Binary "block vs. thrust" accuracy (classes 2,4,5,7,9,10 vs. 0,1,3,6,8,11)
- Train/validation gap

---

## Dependencies Between Steps

```
Step 1 (PureGCN model) ──────┐
                              ├── Step 4 (training script) ── Step 6 (train)
Step 2 (normalize features) ───┤
                              ├── Step 5 (regenerate .pt)
Step 3 (dynamic edges) ──────┘
```

All steps 1–3 are independent and can be developed in parallel. Step 4 depends on 1. Step 5 depends on 2 and 3. Step 6 depends on 4 and 5.

## Estimated Effort
| Task | Time |
|------|------|
| Create `PureGCN` model | 30 min |
| Add person-normalized node features | 1 hour |
| Add dynamic stick edges | 30 min |
| Create `4f_train_pure_gcn.py` | 1 hour |
| Regenerate `.pt` files | 30 min |
| Train + evaluate (3 seeds) | 2–3 hours |
| **Total** | **~5 hours** |

## Success Criteria
- **Val accuracy ≥ 60%** (up from 57.6%)
- **Left/right recall ≥ 40%** (up from ~0–7%)
- **Train/val gap ≤ 15%** (down from ~35%)

## Risk: What If Plan A Alone Doesn't Hit 60%?

If Plan A achieves only 58–59%, it still provides a **better foundation** than the current HybridGCN. At that point:
- Apply **Plan E** (radical regularization) on top of the PureGCN architecture
- The combination of person-normalized features + smaller model + Focal Loss + DropEdge should push past 60%
- **Plan B** (hand landmarks) remains the nuclear option if A + E plateau below 65%
