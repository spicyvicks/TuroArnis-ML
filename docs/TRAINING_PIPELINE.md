# Arnis Pose Classification — Training Pipeline Documentation

**Project:** Graph Convolutional Network (GCN) for Arnis martial arts pose classification  
**Architecture:** HybridGCN V2 with node features + global hybrid features  
**Primary Issue:** Models misclassify correct poses due to train-inference gaps  

**Document Purpose:** Complete audit trail of the training pipeline so you can identify bugs, trace data flow, and verify consistency between training and production.

---

## Executive Summary

The training pipeline creates a pose classification system in 4 main steps:

1. **Extract reference features** from ideal poses to create templates
2. **Generate training features** (node + hybrid) from all training images
3. **Train HybridGCN model** with GCN for nodes + MLP for global context
4. **Evaluate** with per-class accuracy and confusion matrices

### Critical Finding

**Production inference uses DIFFERENT feature extraction than training** — this is the PRIMARY cause of misclassification.

| Pipeline | Stick Correction | Impact |
|----------|------------------|--------|
| Training | Method 4 (sophisticated) | Correct features |
| Production | None (raw YOLO) | Wrong features → misclassification |

---

## Step-by-Step Pipeline

### Step 0: Dataset Preparation (Not Automated)

**Manual setup required:**

```
dataset/
├── train/
│   ├── front/
│   │   ├── left_chest_thrust_correct/*.jpg
│   │   ├── left_elbow_block_correct/*.jpg
│   │   └── ... (13 classes)
│   ├── left/
│   │   └── ... (same classes)
│   └── right/
│       └── ... (same classes)
└── test/
    └── ... (same structure)
```

**Data collection notes:**
- Each class needs reference poses in `reference_poses/{viewpoint}/{class_name}/`
- Training images organized by viewpoint and class
- Augmentation happens in Step 2 (3x copies: 1 original + 2 flipped)

---

### Step 1: Extract Reference Features

**Script:** `hybrid_classifier/1_extract_reference_features.py` (347 lines)

**Purpose:** Compute feature templates (mean, std) from reference poses for each class/viewpoint combination.

**Input:**
- `reference_poses/{viewpoint}/{class_name}/*.jpg`
- YOLO stick detector: `runs/pose/arnis_stick_detector/weights/best.pt`

**Output:**
- `hybrid_classifier/feature_templates.json`
- `hybrid_classifier/feature_templates_mirrored.json` (for flipped poses)

**Key Functions:**

| Function | Lines | Purpose |
|----------|-------|---------|
| `apply_stick_method4_correction()` | 71-161 | **CRITICAL:** Sophisticated stick correction |
| `extract_geometric_features()` | 155-230 | Extract 30 geometric features |
| `mirror_features()` | 61-68 | Negate horizontal features for flipped poses |

**⚠️ CRITICAL: Method 4 Stick Correction (lines 71-161)**

```python
def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, 
                                   img_height, world_landmarks, viewpoint=None):
    """
    1. Foreshortening check: Skip correction if stick < 40px (end-on view)
    2. Hand proximity: Determine LEFT/RIGHT from YOLO grip distance to wrists
    3. Pinky snap: Anchor grip to MediaPipe pinky landmark (anatomically stable)
    4. Shin-based length: Use knee→ankle 3D ratio to compute stick length in pixels
    5. Direction preservation: Use raw YOLO direction, only adjust length
    """
```

**Features extracted (30 total):**

| Category | Features |
|----------|----------|
| Joint angles | left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle |
| Heights (relative to hip) | left_wrist_height, right_wrist_height, left_elbow_height, right_elbow_height, stick_tip_height, stick_grip_height |
| Horizontal positions | left_wrist_x, right_wrist_x, stick_tip_x, stick_grip_x |
| Stick orientation | stick_angle, stick_dx, stick_dy, stick_length |
| Expert features | tip_vs_nose, tip_vs_shoulder, tip_vs_hip, r_hand_vs_nose, r_hand_vs_shoulder, r_hand_vs_hip, tip_side, grip_side, foot_stagger, hands_distance |

**Template structure:**
```json
{
  "front_left_chest_thrust_correct": {
    "left_elbow_angle": {"mean": 45.2, "std": 3.1},
    "right_elbow_angle": {"mean": 89.3, "std": 5.2},
    ...
  }
}
```

**Issues identified:**
- Single reference pose per class (lines 287-319) → small std values
- Very small std causes near-zero similarity for valid non-reference poses
- No multiple reference aggregation

---

### Step 2: Generate Node + Hybrid Features

**Script:** `hybrid_classifier/2b_generate_node_hybrid_features.py` (470 lines)

**Purpose:** Extract features for ALL training/test images and save as PyTorch tensors.

**Input:**
- `dataset/{train,test}/{viewpoint}/{class_name}/*.jpg`
- Templates from Step 1: `feature_templates.json`

**Output:**
- `hybrid_classifier/hybrid_features_v3/train_features.pt`
- `hybrid_classifier/hybrid_features_v3/test_features.pt`
- (Or per-viewpoint: `train_features_front.pt`, etc.)

**Key Functions:**

| Function | Lines | Purpose |
|----------|-------|---------|
| `apply_stick_method4_correction()` | 59-161 | **SAME as Step 1** — stick correction |
| `extract_raw_features()` | 163-286 | Get pose + stick + geometric features |
| `compute_hybrid_features()` | 294-318 | Convert to similarity scores using templates |
| `extract_node_features()` | 320-340 | Create 35×6 node feature matrix |
| `process_dataset()` | 379-457 | Parallel processing with multiprocessing |

**Node features (35 nodes × 6 dimensions):**

```python
node_features = [x, y, z, visibility, dist_to_hip_3d, angle_from_hip]
```

Nodes 0-32: MediaPipe pose landmarks  
Node 33: Stick grip  
Node 34: Stick tip

**Hybrid features (30 similarity scores):**

```python
similarity = exp(-0.5 * ((value - mean) / std) ** 2)
```

Gaussian similarity to template mean/std for each geometric feature.

**Data augmentation (built into Step 2):**

```python
# Line 417 in process_dataset()
images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
# + augmented copies with "_aug" suffix
```

Augmentation happens in separate script (0c) — creates:
- 1 non-flipped copy
- 2 flipped copies (horizontal mirror)

**⚠️ Issue: Flipped pose ambiguity**
- Left/right poses become ambiguous when flipped
- `mirror_features()` negates horizontal features (lines 61-68)
- May not properly handle all pose variations

**Output tensor shapes:**
```python
{
  'node_features': torch.Size([N, 35, 6]),    # N samples
  'hybrid_features': torch.Size([N, 30]),      # N samples  
  'labels': torch.Size([N]),                   # class indices
  'viewpoints': list of strings
}
```

---

### Step 2c: Extract Test Features (Alternative)

**Script:** `hybrid_classifier/2c_extract_test_features.py` (similar to 2b)

Used for extracting features from new test images not in original dataset.

---

### Step 3: Train HybridGCN V2

**Script:** `hybrid_classifier/4c_train_hybrid_gcn_v2.py` (442 lines)

**Purpose:** Train specialist models for each viewpoint (front/left/right).

**Input:**
- `.pt` files from Step 2

**Output:**
- `hybrid_classifier/models/model_{viewpoint}.pth`
- `hybrid_classifier/models/history_{viewpoint}.json` (training metrics)

**Architecture: HybridGCN (lines 40-124)**

```
Input: 
  - Node features: 35 nodes × 6 dims (x, y, z, vis, dist_to_hip, angle)
  - Hybrid features: 30 similarity scores
  - Edge index: SKELETON_EDGES (28 bidirectional edges)

Architecture:
  1. Node embedding: 35 nodes → 8-dim learnable embeddings
  2. GCN layers: 3 layers of GCNConv with BatchNorm + ReLU + Dropout
  3. Global pooling: Mean pool node features → graph-level representation
  4. Hybrid MLP: 2-layer MLP processing 30 hybrid features
  5. Fusion: Concatenate GCN output + hybrid MLP output
  6. Classification: Linear layer → 13 class logits
```

**SKELETON_EDGES (lines 31-37):**

```python
SKELETON_EDGES = [
    (11, 12), (12, 11),  # Shoulders (bidirectional)
    (11, 23), (23, 11),  # Left shoulder to hip
    (12, 24), (24, 12),  # Right shoulder to hip
    (23, 24), (24, 23),  # Hips (bidirectional)
    (11, 13), (13, 11),  # Left shoulder to elbow
    (13, 15), (15, 13),  # Left elbow to wrist
    (12, 14), (14, 12),  # Right shoulder to elbow
    (14, 16), (16, 14),  # Right elbow to wrist
    (23, 25), (25, 23),  # Left hip to knee
    (25, 27), (27, 25),  # Left knee to ankle
    (24, 26), (26, 24),  # Right hip to knee
    (26, 28), (28, 26),  # Right knee to ankle
    (15, 33), (33, 15),  # Left wrist to stick grip
    (16, 33), (33, 16),  # Right wrist to stick grip
    (33, 34), (34, 33),  # Stick grip to tip
]
```

**Training config (default):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 150 | Early stopping at patience=20 |
| Learning rate | 0.001 | Adam optimizer |
| Hidden dim | 256 | Can reduce to 128 for less overfitting |
| Dropout | 0.5 | Should increase to 0.7 |
| Batch size | 32 | With drop_last=True |
| Node embedding | 8 dims | May be too small for 35 nodes |
| Augmentation | True | Enabled by default |

**Class imbalance handling (lines 164-170):**

```python
def compute_class_weights(train_graphs):
    labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = len(labels)
    weights = total / (len(CLASS_NAMES) * class_counts + 1e-6)
    return torch.FloatTensor(weights)
```

**Overfitting detection (from history files):**

```json
{
  "epoch": 12,
  "train_acc": 67.3,
  "test_acc": 46.3,
  "gap": 21.0  // > 20% threshold → early stop
}
```

**Issues identified:**
- Left view model severely overfitted (21% gap, stopped at epoch 12)
- Dropout 0.5 insufficient for 256 hidden dim with ~1,871 training images
- BatchNorm on small batches (32) → inference variance

---

### Step 4: Evaluate Model

**Script:** `hybrid_classifier/4d_evaluate_model.py` (577 lines)

**Purpose:** Generate comprehensive evaluation metrics and plots.

**Input:**
- Trained model: `models/model_{viewpoint}.pth`
- Test features: `hybrid_features_v3/test_features.pt`

**Output plots:**

| Plot | Purpose |
|------|---------|
| Confusion matrix | Per-class prediction patterns |
| Threshold sensitivity | Accuracy vs confidence threshold |
| Cross-threshold heatmap | Which classes fail at which thresholds |
| Per-class accuracy | Identify underperforming classes |
| Training history | Loss curves, overfitting detection |
| Ground truth distribution | Class balance visualization |

**Key metrics tracked:**
- Overall accuracy
- Per-class precision/recall/F1
- Per-class accuracy (from confusion matrix)
- Confidence threshold sensitivity

---

### Additional Analysis Scripts

| Script | Purpose |
|--------|---------|
| `4a_train_hybrid_gcn_baseline.py` | Baseline GCN (no hybrid features) for comparison |
| `4b_train_hybrid_gcn_gat.py` | GAT (Graph Attention) variant |
| `4d_train_hybrid_gat_v2.py` | GAT v2 with attention mechanism |
| `5_analyze_hybrid_gcn.py` | Model interpretability analysis |
| `6_plot_training_history.py` | Visualize training curves |
| `7_cross_evaluate_viewpoints.py` | Test model on other viewpoints |
| `8_compare_models.py` | Compare specialist vs merged models |
| `visualize_expert_features.py` | Visualize what features model uses |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Images (dataset/)
    │
    ▼
┌────────────────────────────────┐
│ Step 1: Extract Reference      │  ← Method 4 stick correction APPLIED
│ Features                       │  ← Creates templates (mean/std)
│                                │
│ Input: reference_poses/        │
│ Output: feature_templates.json │
└────────────────────────────────┘
    │
    ▼
┌────────────────────────────────┐
│ Step 2: Generate Node +       │  ← Method 4 stick correction APPLIED
│ Hybrid Features                │  ← Creates tensors for training
│                                │
│ Input: dataset/                │
│ Output: *_features.pt          │
└────────────────────────────────┘
    │
    ▼
┌────────────────────────────────┐
│ Step 3: Train HybridGCN V2     │  ← Expects corrected features
│                                │
│ Input: .pt files              │
│ Output: model_{view}.pth       │
└────────────────────────────────┘
    │
    ▼
┌────────────────────────────────┐
│ Step 4: Evaluate               │  ← Validates on corrected test features
│                                │
│ Output: Plots + metrics        │
└────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION (Inference)                             │
└─────────────────────────────────────────────────────────────────────────────┘

Camera Feed / Image
    │
    ▼
┌────────────────────────────────┐
│ feature_extraction.py          │  ← Method 4 stick correction NOT APPLIED
│                                │  ← Raw YOLO output only
│                                │
│ Lines 69-80: Raw YOLO →        │
│ no correction                  │
└────────────────────────────────┘
    │
    ▼
┌────────────────────────────────┐
│ inference_realtime_gcn.py      │  ← Uses old SpatialGCN (not HybridGCN)
│                                │  ← Ignores 50% of hybrid features!
│                                │
│ Lines 54-60: SpatialGCN        │
│ (not HybridGCN)                │
└────────────────────────────────┘
    │
    ▼
Prediction (often WRONG)
```

---

## Training vs Production Comparison

### Feature Extraction

| Aspect | Training | Production | Issue |
|--------|----------|------------|-------|
| **Stick correction** | Method 4 (full) | None | **CRITICAL** |
| YOLO detector | YOLOv8 | YOLOv8 | ✓ Same |
| MediaPipe pose | Same | Same | ✓ Same |
| Templates | From corrected poses | Same file | ✓ Same |
| Foreshortening check | 40px threshold | Missing | Minor |
| Pinky snap | Applied | Missing | Major |
| Shin-based length | Applied | Missing | Major |

### Model Architecture

| Aspect | Training | Production | Issue |
|--------|----------|------------|-------|
| Model class | HybridGCN | SpatialGCN | **CRITICAL** |
| Hybrid features | Used (50% of model) | Ignored | **CRITICAL** |
| Node features | 35 nodes × 6 dims | 35 nodes × 3 dims | Major |
| SKELETON_EDGES | 28 edges | 30 edges (extra head) | Moderate |

### Critical Mismatches (Root Causes of Misclassification)

1. **Feature extraction mismatch** (lines 59-161 in training, absent in production)
   - Training: Stick length normalized to body proportions
   - Production: Stick length = raw YOLO output
   - Result: Different feature vectors for same pose

2. **Wrong model in inference** (SpatialGCN vs HybridGCN)
   - Training optimizes for hybrid + node features
   - Inference only uses node features
   - Result: 50% of model capacity unused

3. **Skeleton edge mismatch** (28 vs 30 edges)
   - Different graph structure
   - Potential silent feature corruption

---

## Known Issues & Suspect Areas

### Critical (Confirmed)

| # | Issue | Evidence | File:Lines |
|---|-------|----------|------------|
| 1 | Feature extraction mismatch | Production uses raw YOLO | `feature_extraction.py:69-80` vs `2b_generate:59-161` |
| 2 | Wrong model class | SpatialGCN vs HybridGCN | `inference_realtime_gcn.py:54-60` vs `4c_train:40-124` |
| 3 | Skeleton edge mismatch | 30 edges vs 28 edges | `inference:30-37` vs `4c_train:31-37` |

### High (Strong Evidence)

| # | Issue | Evidence | Impact |
|---|-------|----------|--------|
| 4 | Missing hybrid features in inference | SpatialGCN has no hybrid pathway | 50% model capacity lost |
| 5 | Class imbalance | 6.7:1 ratio (248 vs 37 samples) | Bias toward right-side blocks |
| 6 | Left view overfitting | 21% train-test gap | Left poses unreliable |

### Medium (Needs Investigation)

| # | Issue | Evidence | File |
|---|-------|----------|------|
| 7 | Template std too tight | Single reference per class | `1_extract:287-319` |
| 8 | BatchNorm instability | batch_size=32, test<200 | `4c_train` |
| 9 | Embedding dim too small | 8 dims for 35 nodes | `4c_train:55` |
| 10 | Augmentation artifacts | Flipping left/right poses | `0c_augment` |

---

## Audit Checklist

### Files to Examine

- [ ] `deployment_package/src/feature_extraction.py` — Compare to training
- [ ] `inference_realtime_gcn.py` — Check model class and edges
- [ ] `hybrid_classifier/2b_generate_node_hybrid_features.py:59-161` — Stick correction logic
- [ ] `hybrid_classifier/4c_train_hybrid_gcn_v2.py:31-37` — Training skeleton edges
- [ ] `hybrid_classifier/1_extract_reference_features.py:287-319` — Template computation

### Tests to Write

- [ ] `test_feature_extraction_parity.py` — Verify training vs inference produce identical features
- [ ] `test_stick_correction.py` — Geometric validation of stick length ratios
- [ ] `test_class_balance.py` — Verify minimum samples per class before training
- [ ] `test_skeleton_edges.py` — Verify edge index consistency

### Verifications to Run

- [ ] Extract features from same image through both pipelines → `np.allclose()`
- [ ] Check model loading — does inference load HybridGCN weights into SpatialGCN?
- [ ] Count training samples per class → identify underrepresented classes
- [ ] Plot training history → identify overfitting patterns

---

## Appendix A: Method 4 Stick Correction Algorithm

Full algorithm from `2b_generate_node_hybrid_features.py:59-161`:

```python
def apply_stick_method4_correction(raw_grip_px, raw_tip_px, kpts, img_width, 
                                   img_height, world_landmarks, viewpoint=None):
    """
    Apply Stick Detection Method 4 (Updated): Adaptive Stick Correction
    
    1. FORESHORTENING CHECK
       - If raw stick length < 40px (end-on view)
       - Skip correction to avoid wild snaps
       - Return raw YOLO output
    
    2. HAND PROXIMITY
       - Compute distance from YOLO grip to left_wrist and right_wrist
       - Choose hand_label = "RIGHT" if closer to right_wrist else "LEFT"
    
    3. PINKY SNAP
       - Get pinky landmark index (18 for RIGHT, 17 for LEFT)
       - Snap grip anchor to pinky position in pixels
       - This is anatomically stable vs raw YOLO grip
    
    4. SHIN-BASED LENGTH (Unified for all views)
       - Get left/right knee and ankle in 3D world coordinates
       - Compute average shin length in meters
       - Compute average shin length in pixels
       - stick_px = shin_px × (STICK_LENGTH_M / shin_m)
       - Clamp to 2.5× torso (sanity check)
    
    5. DIRECTION PRESERVATION
       - Use raw YOLO grip→tip direction vector
       - Normalize to unit vector
       - Project corrected_tip = grip + direction_unit × stick_px
    
    Returns: corrected_grip_px, corrected_tip_px
    """
    STICK_LENGTH_M = 0.71  # Standard Arnis stick length
```

**Key parameters:**
- `FORESHORTEN_THRESHOLD_PX = 40`
- `STICK_LENGTH_M = 0.71`
- Clamping: `stick_px = min(stick_px, avg_torso_px * 2.5)`

---

## Appendix B: Class Distribution

From `CONCERNS.md` analysis (front view training):

| Class | Count | % of Total | Risk |
|-------|-------|------------|------|
| right_elbow_block_correct | 248 | 13.3% | ✓ Well represented |
| neutral | 200 | 10.7% | ✓ Adequate |
| left_knee_block_correct | 196 | 10.5% | ✓ Adequate |
| ... | ... | ... | ... |
| crown_thrust_correct | 42 | 2.2% | ⚠️ Underrepresented |
| left_eye_thrust_correct | 41 | 2.2% | ⚠️ Underrepresented |
| left_elbow_block_correct | 39 | 2.1% | ⚠️ Underrepresented |
| left_chest_thrust_correct | 37 | 2.0% | ⚠️ Severely underrepresented |

**Ratio:** 6.7:1 between most/least represented classes

**Impact:** Model biased toward right-side blocks, left thrusts systematically misclassified.

---

## Appendix C: Skeleton Edge Definitions

### Training (HybridGCN) — 28 edges

```python
# hybrid_classifier/4c_train_hybrid_gcn_v2.py:31-37
SKELETON_EDGES = [
    # Shoulders (bidirectional)
    (11, 12), (12, 11),
    # Torso
    (11, 23), (23, 11),  # L shoulder ↔ L hip
    (12, 24), (24, 12),  # R shoulder ↔ R hip
    (23, 24), (24, 23),  # Hips ↔
    # Left arm
    (11, 13), (13, 11),  # Shoulder ↔ elbow
    (13, 15), (15, 13),  # Elbow ↔ wrist
    # Right arm
    (12, 14), (14, 12),
    (14, 16), (16, 14),
    # Left leg
    (23, 25), (25, 23),  # Hip ↔ knee
    (25, 27), (27, 25),  # Knee ↔ ankle
    # Right leg
    (24, 26), (26, 24),
    (26, 28), (28, 26),
    # Stick connections
    (15, 33), (33, 15),  # L wrist ↔ stick grip
    (16, 33), (33, 16),  # R wrist ↔ stick grip
    (33, 34), (34, 33),  # Grip ↔ tip
]
```

### Production (SpatialGCN) — 30 edges

```python
# inference_realtime_gcn.py:30-37 (inferred from CONCERNS.md)
SKELETON_EDGES = [
    # ... same as above ...
    # EXTRA HEAD EDGES (not in training!)
    (3, 7), (7, 3),  # Left ear ↔ left eye (or similar)
    # ... stick edges same ...
]
```

**⚠️ Edge mismatch:** Production adds 2 head edges not present in training.

---

## Summary: Root Cause of "Correct Poses Misclassified"

The primary cause is a **train-inference feature extraction mismatch**:

1. **Training** uses sophisticated stick correction (Method 4)
2. **Production** uses raw YOLO output
3. **Result:** Systematic feature vector differences → misclassification

Secondary causes:
1. **Left view underfitting** — insufficient data + early stopping
2. **Class imbalance bias** — model favors right-side blocks
3. **Missing hybrid features** — inference ignores 50% of model capacity

**Immediate action:** Fix stick correction in production before retraining any models.

---

*Documentation created: 2025-04-20*  
*Pipeline version: HybridGCN V2*  
*Based on code audit of: hybrid_classifier/ (15 scripts)*
