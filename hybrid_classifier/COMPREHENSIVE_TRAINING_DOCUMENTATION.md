# Comprehensive Documentation: Hybrid Features V2 Training

## Executive Summary

The Hybrid Features V2 system is a specialized machine learning pipeline that achieves **82.6% (Front), 77.2% (Left), and 83.7% (Right)** accuracy on Arnis stance classification. This represents a significant improvement over the previous RF/XGB approach (50.4% accuracy) by combining Graph Neural Networks (GCN) with expert-engineered geometric features.

**Key Achievement**: The system uses **three specialist viewpoint models** (Front, Left, Right) trained independently, which are selected at inference time based on the camera's perspective.

---

## Table of Contents

1. [The 3 Viewpoint Models Explained](#the-3-viewpoint-models-explained)
2. [Dataset Transformation Pipeline](#dataset-transformation-pipeline)
3. [Training Methodology](#training-methodology)
4. [Performance Analysis](#performance-analysis)
5. [Architecture Deep Dive](#architecture-deep-dive)

---

## The 3 Viewpoint Models Explained

### Overview

The system trains **three separate specialist models**, one for each camera viewpoint:

| Viewpoint | Accuracy | Purpose | File |
|-----------|----------|---------|------|
| **Front** | 82.6% | Camera facing user directly | `hybrid_gcn_v2_front.pth` |
| **Left** | 77.2% | Camera on user's left side | `hybrid_gcn_v2_left.pth` |
| **Right** | 83.7% | Camera on user's right side | `hybrid_gcn_v2_right.pth` |

**Why 3 Models Instead of 1?**

The same Arnis stance looks **geometrically different** from different camera angles:

```
Front View:              Left View:              Right View:
    O                      O                       O
   /|\                     |\                     /|
  / | \                    | \                   / |
   | |                     |  \                 /  |
  /   \                    |   \               /   |
User facing you          User's left side     User's right side
```

**Key Insight**: A single model trying to learn all viewpoints simultaneously gets confused by the geometric variations. Specialist models achieve higher accuracy by learning viewpoint-specific patterns.

### Per-Class Performance Breakdown

#### Front View Model (82.6% overall)

| Class | Accuracy | Notes |
|-------|----------|-------|
| `solar_plexus_thrust_correct` | 93.8% | Excellent - clear frontal signature |
| `right_eye_thrust_correct` | 93.3% | High stick visibility from front |
| `right_temple_block_correct` | 93.3% | Clear arm positioning |
| `left_temple_block_correct` | 89.5% | Good distinguishability |
| `left_chest_thrust_correct` | 86.7% | Slight confusion with right variants |
| `left_eye_thrust_correct` | 82.4% | Some confusion with crown thrust |
| `crown_thrust_correct` | 80.0% | Challenging - high arm position |
| `left_knee_block_correct` | 78.6% | Lower body occlusions |
| `right_knee_block_correct` | 77.8% | Lower body occlusions |
| `right_chest_thrust_correct` | 71.4% | Most confused class |
| `left_elbow_block_correct` | 71.4% | Block vs Thrust confusion |
| `right_elbow_block_correct` | 66.7% | Lowest accuracy - elbow blocks look similar |

**Confusion Pattern**: Elbow blocks (especially right) are frequently confused with chest thrusts because from the front view, the arm positioning looks geometrically similar.

#### Left View Model (77.2% overall)

| Class | Accuracy | Notes |
|-------|----------|-------|
| `right_eye_thrust_correct` | 100% | Perfect - stick clearly visible |
| `left_knee_block_correct` | 92.3% | Excellent from left angle |
| `left_chest_thrust_correct` | 81.8% | Good performance |
| `right_chest_thrust_correct` | 81.8% | Good performance |
| `left_eye_thrust_correct` | 83.3% | Clear view from left |
| `solar_plexus_thrust_correct` | 81.8% | Consistent |
| `right_elbow_block_correct` | 78.6% | Better than front view |
| `left_temple_block_correct` | 73.3% | Some confusion |
| `right_temple_block_correct` | 75.0% | Moderate |
| `right_knee_block_correct` | 63.6% | Occluded from left |
| `left_elbow_block_correct` | 55.6% | Challenging - arm occlusion |
| `crown_thrust_correct` | 53.8% | **Lowest** - arm blocks view |

**Key Observation**: From the left view, the **right side of the body** is more visible and achieves higher accuracy, while the left side (especially crown thrust) is often occluded by the user's own arm.

#### Right View Model (83.7% overall - Best Performer!)

| Class | Accuracy | Notes |
|-------|----------|-------|
| `left_knee_block_correct` | 100% | Perfect - left side visible |
| `solar_plexus_thrust_correct` | 100% | Perfect - clear view |
| `left_eye_thrust_correct` | 90.0% | Excellent |
| `right_eye_thrust_correct` | 90.9% | Excellent |
| `crown_thrust_correct` | 88.9% | Much better than left view |
| `right_elbow_block_correct` | 81.8% | Good visibility |
| `left_chest_thrust_correct` | 81.8% | Consistent |
| `right_knee_block_correct` | 81.8% | Good from right angle |
| `left_elbow_block_correct` | 75.0% | Moderate |
| `left_temple_block_correct` | 75.0% | Moderate |
| `right_temple_block_correct` | 70.0% | Some confusion |
| `right_chest_thrust_correct` | 70.0% | **Lowest** - body occlusion |

**Key Insight**: The right view achieves the highest overall accuracy (83.7%) because most users are **right-handed**, making the right-side movements more visible and distinct from this angle.

### Ensemble Strategy

During inference, the system selects the appropriate model based on the known camera viewpoint:

```python
if viewpoint == "front":
    model = load_model("hybrid_gcn_v2_front.pth")
elif viewpoint == "left":
    model = load_model("hybrid_gcn_v2_left.pth")
elif viewpoint == "right":
    model = load_model("hybrid_gcn_v2_right.pth")

prediction = model.predict(features)
```

**Alternative**: A "merged" model can be trained on all viewpoints combined (see `--merged` flag in training script), achieving ~75-78% accuracy - simpler deployment but lower performance.

---

## Dataset Transformation Pipeline

### Step 1: Reference Feature Extraction

**Script**: `1_extract_reference_features.py`

**What it does**:
- Takes **5 high-quality reference images per class per viewpoint** (195 total: 13 classes × 3 viewpoints × 5 images)
- Extracts geometric "ideal" templates
- Computes mean (μ) and standard deviation (σ) for each feature

**Reference Images Structure**:
```
reference_poses/
├── front/
│   ├── crown_thrust_correct/        ← 5 perfect front-view crown thrusts
│   ├── left_chest_thrust_correct/   ← 5 perfect front-view left chest thrusts
│   └── ... (13 classes)
├── left/
│   └── ... (same 13 classes)
└── right/
    └── ... (same 13 classes)
```

**Features Extracted**:
- **Angles**: Elbow angles, shoulder angles, knee angles
- **Relative Heights**: Wrist/shoulder/hip positions
- **Stick Metrics**: Length, angle, grip-tip distance
- **Body Proportions**: Torso ratios, limb lengths

**Output**: `feature_templates.json` containing statistics for each class/viewpoint combination

### Step 2: Node-Specific Feature Generation

**Script**: `2b_generate_node_hybrid_features.py`

This is where the **magic happens** - transforming raw images into the graph format the GCN expects.

#### A. Preprocessing Pipeline

**Input**: Raw image from `dataset_split/train/` or `dataset_split/test/`

**Processing**:

1. **Stick Detection (YOLOv8)**
   - Detects Arnis stick bounding box
   - Extracts grip and tip keypoints
   - Confidence score for detection quality

2. **Pose Estimation (MediaPipe)**
   - Extracts 33 body landmarks (x, y, z, visibility)
   - Normalized coordinates (0-1 range)
   - 3D world coordinates for depth

3. **Viewpoint Detection**
   - Parse folder name to determine viewpoint
   - Critical for selecting correct correction method

#### B. Stick Correction (Method 4 - Viewpoint-Adaptive)

**Problem**: YOLO stick detection has errors:
- Grip/tip points swapped
- Wrong length (too short/long)
- Incorrect angle

**Solution**: Viewpoint-adaptive correction using body proportions

```python
# For FRONT view:
- Length: 1.5x torso (accounts for foreshortening)
- Direction: Blend YOLO (40%) + hand direction (60%)
- Anchor: MediaPipe wrist (more stable than YOLO grip)

# For LEFT/RIGHT views:
- Length: Calculated from 3D world landmarks
- Direction: Based on forearm extension
- Correction: Use MediaPipe arm pose for validation
```

**Key Innovation**: Different correction strategies for different viewpoints because the stick appears foreshortened differently from each angle.

#### C. Graph Construction

**Nodes (35 total)**:
- **0-32**: MediaPipe body landmarks (33 nodes)
- **33**: Stick grip point
- **34**: Stick tip point

**Node Features (6 dimensions per node)**:
```python
node_features = [
    x,                    # Normalized x coordinate (0-1)
    y,                    # Normalized y coordinate (0-1)
    z,                    # Depth coordinate
    visibility,           # MediaPipe confidence (0-1)
    dist_to_hip_3d,       # Euclidean distance to hip center
    angle_from_hip        # 2D angle relative to hip center
]
# Shape: [35, 6]
```

**Edges (30 edges)**:
```python
SKELETON_EDGES = [
    # Body connectivity (bilateral)
    (11, 12), (12, 11),  # Shoulders
    (11, 23), (23, 11),  # Left shoulder to hip
    (12, 24), (24, 12),  # Right shoulder to hip
    (23, 24), (24, 23),  # Hips
    (11, 13), (13, 11),  # Left shoulder to elbow
    (13, 15), (15, 13),  # Left elbow to wrist
    (12, 14), (14, 12),  # Right shoulder to elbow
    (14, 16), (16, 14),  # Right elbow to wrist
    (23, 25), (25, 23),  # Left hip to knee
    (25, 27), (27, 25),  # Left knee to ankle
    (24, 26), (26, 24),  # Right hip to knee
    (26, 28), (28, 26),  # Right knee to ankle
    
    # Stick connections
    (15, 33), (33, 15),  # Left wrist to grip
    (16, 33), (33, 16),  # Right wrist to grip
    (33, 34), (34, 33),  # Grip to tip
]
```

#### D. Hybrid Features (Global Context)

While the GCN processes local node relationships, the **hybrid features** provide global geometric context by comparing the pose to reference templates.

**Calculation**:
```python
# For each geometric feature:
similarity = exp(-0.5 * ((value - μ) / σ)²)

# Example:
# If current elbow angle = 95°
# Reference mean (μ) = 90°
# Reference std (σ) = 10°
# Similarity = exp(-0.5 * ((95-90)/10)²) = exp(-0.125) = 0.88
```

**Features Compared (20-30 dimensions)**:
- Joint angles (elbow, shoulder, knee)
- Relative heights (wrist, hand, stick tip)
- Stick orientation (angle, direction vectors)
- Body symmetry ratios
- Distance metrics (hand-to-head, stick-to-torso)

**Output Format**:
```python
{
    'node_features': torch.Tensor [N, 35, 6],      # N samples, 35 nodes, 6 features
    'hybrid_features': torch.Tensor [N, 25],       # N samples, 25 similarity scores
    'labels': torch.Tensor [N],                     # N samples, class indices (0-11)
    'viewpoints': List[str]                         # N samples, ['front', 'left', 'right']
}
```

#### E. Viewpoint-Specific Dataset Creation

The script generates **separate datasets** for each viewpoint:

```
hybrid_features_v2/
├── train_features_front.pt      # Front view training data
├── test_features_front.pt       # Front view test data
├── train_features_left.pt       # Left view training data
├── test_features_left.pt        # Left view test data
├── train_features_right.pt      # Right view training data
└── test_features_right.pt       # Right view test data
```

Each file contains PyTorch tensors ready for GCN training.

### Data Augmentation (Training Time)

During training, the following augmentations are applied:

1. **Random Scaling** (0.85x - 1.15x)
   - Simulates different camera distances
   - Scales x, y, z coordinates and distance features

2. **Coordinate Jitter** (±2% noise)
   - Adds Gaussian noise to (x, y, z)
   - Improves robustness to detection errors

**Example**:
```python
scale = 0.85 + (0.3 * random())  # 0.85 to 1.15
x_augmented = x * scale + noise
```

---

## Training Methodology

### Architecture: HybridGCN Model

**File**: `4c_train_hybrid_gcn_v2.py`

The model combines two parallel processing streams:

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Graph                          │
│  Node Features: [35, 6]    Hybrid Features: [25]            │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
         ┌─────────▼──────────┐  ┌────────▼────────┐
         │   GCN Stream       │  │  MLP Stream     │
         │  (Spatial)         │  │  (Expert)       │
         │                    │  │                 │
         │ Node Embedding     │  │ FC1: 25→256     │
         │ (35 nodes × 8 dim) │  │ BatchNorm       │
         │         +          │  │ ReLU            │
         │ Node Features      │  │ Dropout(0.5)    │
         │ (35 nodes × 6 dim) │  │                 │
         │         ↓          │  │ FC2: 256→256    │
         │   Concatenate      │  │ BatchNorm       │
         │ (35 nodes × 14 dim)│  │ ReLU            │
         │         ↓          │  │ Dropout(0.5)    │
         │  GCNConv Layer 1   │  └────────┬────────┘
         │  14 → 256 channels │           │
         │  BatchNorm         │           │
         │  ReLU              │           │
         │  Dropout(0.5)      │           │
         │         ↓          │           │
         │  GCNConv Layer 2   │           │
         │  256 → 256 channels│           │
         │  BatchNorm         │           │
         │  ReLU              │           │
         │  Dropout(0.5)      │           │
         │         ↓          │           │
         │  GCNConv Layer 3   │           │
         │  256 → 256 channels│           │
         │  BatchNorm         │           │
         │  ReLU              │           │
         │  Dropout(0.5)      │           │
         │         ↓          │           │
         │ Global Mean Pool   │           │
         │  (35 → 1 node)     │           │
         │  [batch, 256]      │           │
         └─────────┬──────────┘           │
                   │                      │
                   └──────────┬───────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Fusion Layer     │
                    │                    │
                    │ Concatenate:       │
                    │ [256 (GCN) +       │
                    │  256 (MLP)]        │
                    │         ↓          │
                    │ FC: 512 → 256      │
                    │ BatchNorm          │
                    │ ReLU               │
                    │ Dropout(0.5)       │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Classification Head│
                    │ FC: 256 → 12       │
                    │ Softmax            │
                    └────────────────────┘
```

**Key Components**:

1. **Node Identity Embeddings (8-dim)**
   - Learnable embeddings for each of the 35 nodes
   - Helps model distinguish wrist (node 15) from elbow (node 13)
   - Added to geometric features: [x, y, z, vis, dist, angle] + [emb_0...emb_7]

2. **3-Layer GCN**
   - Each layer: GCNConv with 256 hidden channels
   - BatchNorm for training stability
   - ReLU activation
   - Dropout(0.5) for regularization

3. **Hybrid MLP (2 layers)**
   - Encodes geometric similarity scores
   - Same 256-dim hidden size for balanced fusion

4. **Fusion Layer**
   - Concatenates GCN output (256) + Hybrid output (256) = 512
   - Projects back to 256 for classification

**Parameter Count**: ~500K parameters (lightweight for CPU training)

### Training Configuration

```python
# Hyperparameters
epochs = 150
learning_rate = 0.001
hidden_dim = 256
dropout = 0.5
num_layers = 3
batch_size = 32
optimizer = Adam(weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=15, factor=0.5)

# Loss Function
criterion = CrossEntropyLoss(class_weights)
# Class weights compensate for imbalanced dataset

# Early Stopping
patience = 20 epochs
overfitting_threshold = 20% gap (train - test)
```

### Training Process

**For Each Viewpoint**:

```bash
# Train Front specialist
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint front

# Train Left specialist
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint left

# Train Right specialist
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint right
```

**Training Loop**:

1. **Epoch 1-150**:
   - Train on viewpoint-specific data
   - Apply data augmentation (scaling + jitter)
   - Monitor train/test accuracy gap

2. **Early Stopping Triggers**:
   - No improvement for 20 epochs
   - Overfitting gap > 20% (train >> test)

3. **Best Model Selection**:
   - Save checkpoint when test accuracy improves
   - Keep best model across all epochs

**Typical Training Time**: 30-60 minutes per viewpoint on Intel Core 7 CPU

### Class Imbalance Handling

Some stances appear more frequently in training data. Class weights are computed:

```python
class_weights = total_samples / (num_classes × class_count)

# Example:
# If crown_thrust appears 100 times out of 1000 total
# weight = 1000 / (12 × 100) = 0.83
# If left_elbow_block appears 50 times
# weight = 1000 / (12 × 50) = 1.67 (higher weight)
```

This penalizes the model more for misclassifying rare classes.

---

## Performance Analysis

### Overall Results Summary

| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Right View** | 83.7% | Best overall, good for right-handed users | Right chest thrust (70%) |
| **Front View** | 82.6% | Balanced, good general performance | Elbow blocks (66-71%) |
| **Left View** | 77.2% | Good for left-side movements | Crown thrust (53.8%), left elbow (55.6%) |
| **Specialist Ensemble** | **81.2% avg** | Highest per-view accuracy | Requires viewpoint knowledge |
| Merged Model (all views) | ~75-78% | Simpler deployment | Lower accuracy |
| Previous RF/XGB | 50.4% | Fast training | Poor generalization |

### Confusion Analysis

**Most Common Errors**:

1. **Elbow Block ↔ Chest Thrust Confusion**
   - Right elbow block (66.7%) confused with right chest thrust
   - Similar arm positioning from front view
   - **Mitigation**: Better stick angle features

2. **Crown Thrust Classification**
   - 53.8% from left view (lowest)
   - 88.9% from right view (much better)
   - Arm occludes view from left side
   - **Mitigation**: Specialist model selection

3. **Knee Blocks**
   - Lower accuracy across all views (77-82%)
   - Lower body often partially occluded
   - **Mitigation**: Better leg feature extraction

### Per-Viewpoint Recommendations

**For Right-Handed Users** (most common):
- **Best camera position**: Right side (83.7% accuracy)
- Provides clearest view of stick arm
- Crown thrust and eye thrust clearly visible

**For Training Applications**:
- **Best camera position**: Front (82.6% accuracy)
- Most natural for user interaction
- Balanced view of all movements

**Avoid**: Pure left-side camera (77.2%)
- Unless user is left-handed
- Many right-side movements occluded

---

## Architecture Deep Dive

### Why Graph Neural Networks?

**Traditional Approach (RF/XGB)**:
```
Image → Extract 110 angles/distances → Table → Classify
Problem: Loses spatial relationships!
```

**Graph Approach (HybridGCN)**:
```
Image → Build skeleton graph → Learn node relationships → Classify
Advantage: Preserves "elbow connects to wrist" structure
```

**Example**: Forward Stance
- RF sees: `left_knee_angle=120°, right_knee_angle=170°, ...`
- GCN sees: `left_knee --connects-to--> left_hip --connects-to--> torso`

The GCN learns that knee angle RELATIVE to hip position matters, not just the absolute angle.

### Why Hybrid Features?

**Pure GCN Problem**: 
- Learns spatial patterns but misses "ideal pose" knowledge
- Doesn't know that 90° elbow = "good chamber"

**Hybrid Solution**:
- GCN learns: "This arrangement of joints looks like Forward Stance"
- Hybrid adds: "This pose is 88% similar to ideal Forward Stance template"
- Combined: More robust classification

### Edge Importance

The skeleton edges encode anatomical constraints:

```python
# Without edges (pure node features):
"Node 15 (wrist) is at position (0.5, 0.3)"

# With edges:
"Node 15 (wrist) is at (0.5, 0.3) AND connects to Node 13 (elbow)"
→ Model learns: wrist position constrained by elbow position
```

This prevents physically impossible predictions (e.g., wrist 2 meters from elbow).

### Node Embeddings Explained

**Problem**: How does the model distinguish left wrist (node 15) from right wrist (node 16) if they have similar coordinates?

**Solution**: Learnable node identity embeddings

```python
# Each node gets a unique 8-dimensional "name tag"
node_15_embedding = [0.2, -0.5, 0.1, 0.8, -0.3, 0.4, 0.0, 0.6]  # Left wrist
node_16_embedding = [-0.1, 0.3, 0.7, -0.2, 0.5, -0.4, 0.2, 0.1]  # Right wrist

# Concatenated with geometric features:
features = [x, y, z, vis, dist, angle, emb_0, emb_1, ..., emb_7]
          # Geometric (6)        + Identity (8) = 14 dimensions
```

During training, the model learns:
- "Node 15 usually has lower x than node 16" (left vs right)
- "Node 15 connected to node 13 means left arm"

### Comparison to Pure Approaches

| Approach | Accuracy | Pros | Cons |
|----------|----------|------|------|
| **Hybrid GCN (this work)** | 77-84% | Best accuracy, uses domain knowledge | Requires viewpoint labels |
| Pure GCN (no hybrid) | ~65-70% | Simple, learns from structure only | Misses "ideal pose" knowledge |
| Pure MLP (hybrid only) | ~60-65% | Fast, uses domain knowledge | Loses spatial relationships |
| Random Forest | 50.4% | Very fast, simple | Poor generalization, overfitting |

---

## File Reference

### Core Training Files

| File | Purpose |
|------|---------|
| `1_extract_reference_features.py` | Extract ideal pose templates from reference images |
| `2b_generate_node_hybrid_features.py` | Convert dataset to graph format with hybrid features |
| `4c_train_hybrid_gcn_v2.py` | Train HybridGCN model (GCN + MLP fusion) |
| `4d_train_hybrid_gat_v2.py` | Alternative: Graph Attention Network variant |

### Model Artifacts

| File | Description |
|------|-------------|
| `hybrid_gcn_v2_front.pth` | Trained front-view specialist model |
| `hybrid_gcn_v2_left.pth` | Trained left-view specialist model |
| `hybrid_gcn_v2_right.pth` | Trained right-view specialist model |
| `feature_templates.json` | Reference pose statistics (μ, σ per class/view) |
| `history_{viewpoint}.json` | Training history (loss, accuracy per epoch) |

### Evaluation Files

| File | Description |
|------|-------------|
| `evaluation_{viewpoint}.json` | Test accuracy, per-class accuracy, confusion matrix |
| `confusion_matrix_{viewpoint}.png` | Visual confusion matrix |
| `per_class_accuracy_{viewpoint}.png` | Bar chart of per-class performance |

---

## Usage Instructions

### 1. Prepare Reference Images

```bash
# Copy 5 high-quality images per class per viewpoint
# Total: 195 images (13 classes × 3 viewpoints × 5 images)
# Into: reference_poses/{front,left,right}/{class_name}/
```

### 2. Extract Reference Templates

```bash
python hybrid_classifier/1_extract_reference_features.py
# Output: hybrid_classifier/feature_templates.json
```

### 3. Generate Graph Features

```bash
python hybrid_classifier/2b_generate_node_hybrid_features.py
# Output: hybrid_classifier/hybrid_features_v2/*.pt
```

### 4. Train Specialist Models

```bash
# Train all three viewpoints
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint front
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint left
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint right

# Or train merged model (simpler but lower accuracy)
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --merged
```

### 5. Evaluate

```bash
# Check evaluation JSON files
python -c "import json; print(json.load(open('hybrid_classifier/evaluation/evaluation_front.json'))['accuracy'])"
# Output: 0.8260869565217391 (82.6%)
```

---

## Conclusion

The Hybrid Features V2 system represents a **major advancement** in Arnis stance classification:

✅ **Accuracy**: 77-84% (vs 50.4% previous)  
✅ **Approach**: Graph Neural Networks + Expert Features  
✅ **Specialization**: 3 viewpoint-specific models  
✅ **Robustness**: Geometric stick correction  
✅ **Deployment**: ONNX-compatible for app integration  

**Next Steps for Improvement**:
1. Collect more left-view training data (currently weakest)
2. Add temporal smoothing for video sequences
3. Implement automatic viewpoint detection
4. Explore attention mechanisms (GAT vs GCN)

**Target**: 85%+ accuracy with automatic viewpoint selection

---

*Documentation generated for TuroArnis ML Project*  
*Hybrid Features V2 - Graph Neural Network Pipeline*
