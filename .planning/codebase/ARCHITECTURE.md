# Architecture

**Analysis Date:** 2026-04-18

## Overall Pattern

**Primary Architecture:** Hybrid ML Pipeline combining Graph Neural Networks with Rule-based Feature Engineering

**Key Characteristics:**
1. **Dual-track approach** - Both GCN and hybrid classifier pipelines coexist
2. **Multi-modal input** - Combines body pose (MediaPipe) + stick detection (YOLO)
3. **Graph-based representation** - Body + stick represented as spatial graph (35 nodes)
4. **Temporal smoothing** - Deque buffer for stable real-time predictions
5. **Viewpoint-aware** - Separate models/weights for front, left, right camera angles

## System Layers

### Layer 1: Data Ingestion & Preprocessing
**Purpose:** Raw image/video → Structured features
**Location:** Root-level scripts (`0_*.py`, `0c_*.py`), `feature_extraction.py`
**Contains:**
- Image format conversion (`0_convert_heic_to_jpg.py`)
- Dataset splitting (`0_organize_and_verify_split.py`)
- Data augmentation (`0c_augment_training_data.py`)
- Image flipping for viewpoint correction (`0c_flip_*.py`)

**Key Files:**
- `0_organize_and_verify_split.py` - Train/test splitting with verification
- `0c_augment_training_data.py` - Albumentations-based augmentation pipeline
- `deployment_package/src/feature_extraction.py` - Unified feature extraction module

**Depends on:** OpenCV, MediaPipe, YOLO
**Used by:** All downstream training and inference pipelines

### Layer 2: Feature Extraction & Graph Construction
**Purpose:** Images → Geometric features → Graph representation
**Location:** `hybrid_classifier/1_*.py`, `2_*.py`, `2b_*.py`, `2c_*.py`
**Contains:**
- Reference pose analysis
- Geometric feature computation (angles, distances, heights)
- Graph construction with skeleton edges
- Hybrid feature generation (similarity scores)

**Key Components:**
```python
# From feature_extraction.py
extract_raw_features()      # → pose_keypoints, stick_keypoints, global_features
compute_hybrid_features()   # → similarity scores vs templates
extract_node_features()     # → per-node 6D features [x, y, z, vis, dist, angle]
```

**Key Files:**
- `hybrid_classifier/1_extract_reference_features.py` - Template extraction from reference images
- `hybrid_classifier/2b_generate_node_hybrid_features.py` - Node features + hybrid features
- `deployment_package/src/feature_extraction.py` - Deployment-optimized extraction

**Depends on:** Layer 1, MediaPipe, YOLO, NumPy
**Used by:** Model training, real-time inference

### Layer 3: Model Training
**Purpose:** Features → Trained classifiers
**Location:** `hybrid_classifier/3_*.py`, `4_*.py`, `models/`
**Contains:**
- Random Forest baseline
- Neural network classifier
- Graph Neural Networks (GCN, GAT, hybrid variants)
- Cross-evaluation and comparison

**Model Architectures:**

**1. SpatialGCN (`models/spatial_gcn.py`):**
```python
SpatialGCN(
    in_channels=3/6/21,    # node features
    hidden_channels=64,     # base hidden dimension
    num_classes=13,         # Arnis stances
    dropout=0.5
)
# Architecture: 4 GCN layers (in→64→128→256→128) → global pool → Linear → 13 classes
```

**2. HybridGCN (`hybrid_classifier/4c_train_hybrid_gcn_v2.py`):**
```python
HybridGCN(
    node_in_channels=6,     # per-node geometric features
    hybrid_in_channels=30,  # global similarity features
    hidden_channels=256,    # wider for fusion
    num_layers=3,
    embedding_dim=8         # learnable node identity
)
# Architecture: GCN branch + MLP branch → fusion → classification
```

**3. Traditional ML (`hybrid_classifier/3_train_classifier.py`):**
- Random Forest (200 trees, max_depth=20)
- MLP (128→64→32 hidden layers)

**Key Files:**
- `hybrid_classifier/4c_train_hybrid_gcn_v2.py` - Optimized Hybrid GCN V2
- `hybrid_classifier/3_train_classifier.py` - Random Forest / Neural Network
- `models/spatial_gcn.py` - Core GCN architecture

**Depends on:** Layer 2, PyTorch, PyTorch Geometric, scikit-learn
**Used by:** Inference pipeline, model evaluation

### Layer 4: Real-time Inference
**Purpose:** Webcam stream → Live predictions
**Location:** `inference_realtime_gcn.py`, `deployment_package/`
**Contains:**
- Frame capture and preprocessing
- Multi-model inference (MediaPipe + YOLO + GCN)
- Temporal smoothing with deque buffer
- Visualization and UI

**Key Class:** `RealtimeGCNInference`
```python
class RealtimeGCNInference:
    - extract_pose_keypoints()    # MediaPipe
    - detect_stick()              # YOLO
    - build_graph()               # 35-node graph
    - predict()                   # GCN forward pass
    - get_stable_prediction()     # Deque averaging
    - draw_results()              # OpenCV annotation
```

**Key Files:**
- `inference_realtime_gcn.py` - Main real-time script
- `deployment_package/src/feature_extraction.py` - Optimized extraction
- `deployment_package/src/model_architecture.py` - Inference-ready HybridGCN

**Depends on:** All upstream layers, OpenCV
**Used by:** End-user application

### Layer 5: Analysis & Evaluation
**Purpose:** Model analysis, error analysis, visualization
**Location:** `hybrid_classifier/5_*.py`, `6_*.py`, `7_*.py`, `8_*.py`, `analyze_errors.py`
**Contains:**
- Confusion matrices
- Per-class accuracy analysis
- Cross-viewpoint evaluation
- Training history plotting

**Key Files:**
- `hybrid_classifier/4d_evaluate_model.py` - Comprehensive evaluation
- `hybrid_classifier/7_cross_evaluate_viewpoints.py` - Viewpoint generalization
- `analyze_errors.py` - Error pattern analysis

## Data Flow

### Training Pipeline Flow
```
dataset_split/
├── train/                           ← Source images
│   └── [class_name]/
└── test/
    └── [class_name]/
         ↓
1_extract_reference_features.py    ← Analyze reference_poses/
    → feature_templates.json       → Statistics per class/viewpoint
         ↓
2b_generate_node_hybrid_features.py
    → hybrid_features_v3/          → .pt files (node + hybrid features)
         ↓
4c_train_hybrid_gcn_v2.py
    → hybrid_classifier/models/    → .pth checkpoint files
```

### Inference Pipeline Flow
```
Webcam Frame (OpenCV)
    ↓
MediaPipe Pose
    → 33 body keypoints [x, y, z, visibility]
    ↓
YOLO Stick Detector
    → 2 stick keypoints [grip, tip] with confidence
    ↓
Feature Extraction
    → 35 nodes with 6D features each
    → 30 global geometric features
    ↓
Graph Construction
    → Data(x=[35, 6], edge_index=[2, 74])  # 74 edges (bidirectional skeleton)
    ↓
HybridGCN Forward Pass
    → 13 class logits
    ↓
Softmax + Temporal Smoothing
    → Stable class prediction
    ↓
OpenCV Rendering
    → Annotated frame with pose, stick, label
```

## Key Abstractions

### Graph Representation
**Abstraction:** Human pose + stick as spatial graph
**Nodes:** 35 total (33 MediaPipe landmarks + 2 stick endpoints)
**Edges:** 74 bidirectional skeleton connections + stick connections
**Node Features:** `[x, y, z, visibility, distance_to_hip, angle_from_hip]` (6D)

**Files:**
- `models/spatial_gcn.py` - Basic GCN implementation
- `deployment_package/src/model_architecture.py` - HybridGCN with node embeddings

### Feature Templates
**Abstraction:** Statistical profile of "ideal" poses per class/viewpoint
**Structure:** `{viewpoint}_{class_name} → {feature: {mean, std, min, max}}`
**Usage:** Convert raw geometric features to similarity scores via Gaussian

**Files:**
- `hybrid_classifier/feature_templates.json`
- `hybrid_classifier/feature_templates_mirrored.json`

### Class Hierarchy
```python
CLASS_NAMES = [
    'crown_thrust_correct',
    'left_chest_thrust_correct', 'right_chest_thrust_correct',
    'left_elbow_block_correct', 'right_elbow_block_correct',
    'left_eye_thrust_correct', 'right_eye_thrust_correct',
    'left_knee_block_correct', 'right_knee_block_correct',
    'left_temple_block_correct', 'right_temple_block_correct',
    'solar_plexus_thrust_correct',
    'neutral'  # optional
]
```

## Entry Points

### Training Entry Points
| Script | Purpose | Key Args |
|--------|---------|----------|
| `hybrid_classifier/1_extract_reference_features.py` | Build feature templates | `--viewpoint`, `--mirrored` |
| `hybrid_classifier/2b_generate_node_hybrid_features.py` | Generate training data | - |
| `hybrid_classifier/4c_train_hybrid_gcn_v2.py` | Train GCN | `--viewpoint`, `--epochs`, `--hidden`, `--merged` |
| `hybrid_classifier/3_train_classifier.py` | Train RF/MLP | `--model_type`, `--viewpoint` |

### Inference Entry Points
| Script | Purpose | Key Args |
|--------|---------|----------|
| `inference_realtime_gcn.py` | Webcam inference | `--model`, `--camera`, `--buffer_size` |
| `test_real_images.py` | Batch inference on images | - |
| `hybrid_classifier/4d_evaluate_model.py` | Model evaluation | - |

### Utility Entry Points
| Script | Purpose |
|--------|---------|
| `visualize_graph.py` | Visualize graph structure |
| `visualize_reference_poses.py` | Display reference pose statistics |
| `visualize_stick.py` | Debug stick detection |

## Error Handling

**Strategy:** Graceful degradation with fallbacks

**Patterns:**
1. **Pose detection failure** → Skip frame, maintain previous prediction
2. **Stick detection failure** → Use default `[0.5, 0.5, 0.0]` (center, zero confidence)
3. **Model loading failure** → Clear error message, exit with code 1
4. **CUDA unavailable** → Automatic CPU fallback

**Example:**
```python
# From inference_realtime_gcn.py
try:
    checkpoint = torch.load(model_path, map_location=self.device)
except FileNotFoundError:
    print(f"Error: Model not found at {model_path}")
    exit(1)

if not results.pose_landmarks:
    return None  # Skip frame gracefully

if len(results[0].boxes) == 0:
    return np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])  # Default stick
```

## Cross-Cutting Concerns

**Logging:** Print-based logging throughout (`print(f"✓ {message}")`)
**Validation:** Argparse validation for all CLI tools
**Authentication:** None (local/offline system)
**Performance:** FPS counter, timing with `cv2.getTickCount()`
**Persistence:** PyTorch `save/load`, JSON for metadata, joblib for sklearn

---

*Architecture analysis: 2026-04-18*
