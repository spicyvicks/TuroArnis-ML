# Codebase Structure

**Analysis Date:** 2026-04-18

## Directory Layout

```
C:\Users\HP\Documents\GitHub\TuroArnis-ML/
├── .planning/              # GSD planning documents
│   └── codebase/          # This analysis documentation
├── .git/                   # Git repository
├── arnis_stick_detector/  # YOLO stick detection model artifacts
│   ├── weights/          # Trained YOLO weights
│   └── args.yaml         # Training configuration
├── baseline_comparison/   # Legacy baseline experiments
├── dataset/              # Raw training images (gitignored)
│   └── [12 class directories]/
├── dataset_graphs/       # Pre-computed PyG datasets (gitignored)
├── dataset_split/        # Train/test split (gitignored)
│   ├── train/
│   └── test/
├── dataset_stick/       # Stick detection dataset (gitignored)
├── debug_output/         # Debug visualizations (gitignored)
├── deployment_package/   # Production deployment code
│   ├── src/
│   │   ├── feature_extraction.py      # Optimized feature extraction
│   │   ├── model_architecture.py       # HybridGCN for inference
│   │   └── feature_templates.json      # Feature statistics
│   └── requirements.txt  # CPU-only dependencies
├── experiments/         # Legacy experiment scripts
├── hybrid_classifier/   # Hybrid ML pipeline (main development)
│   ├── 1_extract_reference_features.py    # Template extraction
│   ├── 2b_generate_node_hybrid_features.py  # Node feature generation
│   ├── 2c_extract_test_features.py         # Test set extraction
│   ├── 2_generate_hybrid_features.py      # Legacy hybrid features
│   ├── 3_train_classifier.py              # RF/MLP training
│   ├── 4c_train_hybrid_gcn_v2.py          # Optimized Hybrid GCN
│   ├── 4d_train_hybrid_gat_v2.py          # GAT variant
│   ├── 4_train_hybrid_gcn.py              # Original hybrid GCN
│   ├── 4a_train_hybrid_gcn_baseline.py    # Baseline comparison
│   ├── 4b_train_hybrid_gcn_gat.py         # GCN+GAT hybrid
│   ├── 4d_evaluate_model.py               # Comprehensive evaluation
│   ├── 5_analyze_hybrid_gcn.py          # Analysis tools
│   ├── 6_plot_training_history.py         # Visualization
│   ├── 7_cross_evaluate_viewpoints.py    # Cross-view testing
│   ├── 8_compare_models.py                # Model comparison
│   ├── feature_templates.json             # Feature statistics
│   ├── feature_templates_mirrored.json    # Mirrored variants
│   ├── hybrid_features_v3/                # Generated features
│   └── models/                            # Trained checkpoints
├── models/              # GCN model zoo (gitignored partially)
│   ├── spatial_gcn.py            # Core GCN architecture
│   ├── gcn_checkpoints/          # Best GCN weights
│   ├── gcn_front/                # Front-view specialist
│   ├── gcn_left/                 # Left-view specialist
│   ├── gcn_right/                # Right-view specialist
│   ├── v*_*/                     # Versioned model experiments
│   └── active_model.json         # Current active model
├── reference_poses/     # Reference images for hybrid features (gitignored)
│   ├── front/          # 13 class subdirectories
│   ├── left/           # 13 class subdirectories
│   └── right/          # 13 class subdirectories
├── reports/            # Analysis reports
├── runs/               # YOLO training runs (gitignored)
├── venv_gcn/          # Virtual environment (gitignored)
├── [root scripts]/     # Data processing and utilities
└── [model files]/      # YOLO weights (.pt files)
```

## Directory Purposes

### Core ML Pipeline

**`hybrid_classifier/` - Main Development Directory**
- **Purpose:** Hybrid rule-based + ML classification pipeline
- **Contains:** Feature extraction, training scripts, models, evaluation
- **Key Files:**
  - `4c_train_hybrid_gcn_v2.py` - Primary training script
  - `1_extract_reference_features.py` - Template generation
  - `4d_evaluate_model.py` - Model evaluation
  - `feature_templates.json` - Statistical templates
- **Generated:** `hybrid_features_v3/`, `models/`

**`models/` - GCN Model Zoo**
- **Purpose:** Graph Neural Network architectures and checkpoints
- **Contains:** Spatial GCN definition, trained weights, experiments
- **Key Files:**
  - `spatial_gcn.py` - Core GCN architecture (98 lines)
  - `gcn_checkpoints/best_model.pth` - Production model
  - `active_model.json` - Model registry

**`deployment_package/` - Production Code**
- **Purpose:** Optimized, dependency-reduced deployment version
- **Contains:** Inference-only code, CPU-optimized dependencies
- **Key Files:**
  - `src/feature_extraction.py` - Unified extraction (218 lines)
  - `src/model_architecture.py` - Inference-ready HybridGCN (111 lines)
  - `requirements.txt` - CPU-only dependencies

### Data Directories (Gitignored)

**`dataset/` - Raw Training Images**
- **Structure:** 12-13 class subdirectories with images
- **Contents:** JPEG/PNG images of Arnis poses
- **Not committed:** Listed in `.gitignore`

**`dataset_split/` - Train/Test Split**
- **Structure:** `train/` and `test/` subdirectories mirroring `dataset/`
- **Created by:** `0_organize_and_verify_split.py`
- **Ratio:** Typically 80/20 train/test

**`dataset_graphs/` - Pre-computed Graphs**
- **Contents:** PyTorch Geometric Data objects
- **Created by:** `create_graph_dataset.py` (not in root listing, likely in experiments/)
- **Purpose:** Cache graph structure for faster training

**`reference_poses/` - Reference Images**
- **Structure:** `front/`, `left/`, `right/` × 13 class subdirectories
- **Contents:** 5 high-quality reference images per class per viewpoint
- **Total:** 195 reference images (5 × 13 × 3)

**`runs/` - YOLO Training Runs**
- **Contains:** YOLO training artifacts, weights, metrics
- **Key:** `runs/pose/arnis_stick_detector/weights/best.pt`

### Configuration & Documentation

**Root Configuration Files:**
| File | Purpose |
|------|---------|
| `.gitignore` | Excludes large files, datasets, models |
| `requirements_gcn.txt` | Development dependencies (CUDA) |
| `GCN_WORKFLOW.md` | GCN training quick-start guide |

**Documentation in `hybrid_classifier/`:**
| File | Purpose |
|------|---------|
| `README.md` | Pipeline overview and workflow |
| `COMPREHENSIVE_TRAINING_DOCUMENTATION.md` | Detailed training guide |
| `STICK_DETECTION_GUIDE.md` | Stick detection methodology |
| `TRAINING_COMMANDS.md` | Command reference |

## Key File Locations

### Entry Points

**Data Preparation:**
| File | Purpose | Lines |
|------|---------|-------|
| `0_organize_and_verify_split.py` | Dataset splitting | ~170 |
| `0c_augment_training_data.py` | Data augmentation | ~200 |
| `0c_flip_reference_poses.py` | Viewpoint correction | ~130 |

**Training:**
| File | Purpose | Lines |
|------|---------|-------|
| `hybrid_classifier/4c_train_hybrid_gcn_v2.py` | Main GCN training | ~442 |
| `hybrid_classifier/3_train_classifier.py` | RF/MLP training | ~148 |
| `hybrid_classifier/1_extract_reference_features.py` | Template extraction | ~347 |

**Inference:**
| File | Purpose | Lines |
|------|---------|-------|
| `inference_realtime_gcn.py` | Real-time webcam inference | ~327 |
| `deployment_package/src/feature_extraction.py` | Production extraction | ~218 |
| `test_real_images.py` | Batch image inference | ~60 |

**Models:**
| File | Purpose | Lines |
|------|---------|-------|
| `models/spatial_gcn.py` | GCN architecture definition | ~98 |
| `deployment_package/src/model_architecture.py` | HybridGCN for inference | ~111 |

### Model Weight Files

| File | Size | Purpose |
|------|------|---------|
| `yolov8n-pose.pt` | 6.8 MB | YOLO pose pre-trained |
| `yolov8n.pt` | 6.5 MB | YOLO base pre-trained |
| `runs/pose/arnis_stick_detector/weights/best.pt` | Variable | Custom stick detector |
| `models/gcn_checkpoints/best_model.pth` | Variable | Best GCN checkpoint |
| `hybrid_classifier/models/hybrid_gcn_v2.pth` | Variable | Hybrid model checkpoint |

## Naming Conventions

### Files

**Prefix Convention (Root Level):**
- `0_*.py` - Data organization and preprocessing scripts
- `0c_*.py` - Data cleaning/correction utilities
- `test_*.py` - Testing and debugging scripts
- `visualize_*.py` - Visualization utilities
- `analyze_*.py` - Analysis and reporting scripts

**Pipeline Convention (`hybrid_classifier/`):**
- `1_*.py` - Step 1: Reference analysis
- `2_*.py` - Step 2: Feature generation
- `3_*.py` - Step 3: Traditional ML training
- `4_*.py` - Step 4: GCN training variants
- `5_*.py` - Step 5: Analysis
- `6_*.py` - Step 6: Visualization
- `7_*.py` - Step 7: Cross-evaluation
- `8_*.py` - Step 8: Comparison

**Versioning (`models/`):**
- `v{NNN}_{timestamp}_{type}` - Versioned experiments
  - Example: `v056_20260209_032034_xgboost`
  - NNN: Sequential version number
  - timestamp: YYYYMMDD_HHMMSS
  - type: Algorithm used

### Directories

**Snake Case:** All directories use lowercase with underscores
- `hybrid_classifier/`
- `dataset_split/`
- `deployment_package/`

**Semantic Naming:**
- `dataset_` prefix - Data storage directories
- `*_features` - Generated feature storage
- `*_classifier/` - Model training code

## Where to Add New Code

### New Pose Class
1. **Add class to CLASS_NAMES lists:**
   - `inference_realtime_gcn.py` line 21
   - `hybrid_classifier/4c_train_hybrid_gcn_v2.py` line 18
   - `deployment_package/src/model_architecture.py` line 11

2. **Add training data:**
   - Create subdirectory in `dataset_split/train/` and `dataset_split/test/`
   - Add reference images to `reference_poses/{viewpoint}/{class_name}/`

3. **Regenerate templates:**
   ```bash
   python hybrid_classifier/1_extract_reference_features.py
   ```

### New Model Architecture
1. **Define in `models/` or `deployment_package/src/`**
2. **Follow existing pattern:**
   - Inherit from `nn.Module`
   - Accept `in_channels`, `hidden_channels`, `num_classes`, `dropout`
   - Use `global_mean_pool` for graph-level output

3. **Add training script in `hybrid_classifier/`:**
   - Use `4{x}_*.py` naming convention
   - Include argparse for hyperparameters
   - Save checkpoint format compatible with inference

### New Feature Type
1. **Add to `feature_extraction.py`:**
   - Root version for development
   - `deployment_package/src/` version for production

2. **Update feature dimension constants:**
   - In training scripts where `node_feat_dim` is calculated
   - In `hybrid_classifier/1_extract_reference_features.py` if template-related

### New Evaluation Metric
1. **Add to `hybrid_classifier/4d_evaluate_model.py`**
2. **Or create new script following `5_*.py` convention**

## Special Directories

**`venv_gcn/` - Virtual Environment**
- Purpose: Isolated Python environment
- Generated: Yes (by user setup)
- Committed: No (gitignored)

**`__pycache__/` - Python Cache**
- Purpose: Compiled Python bytecode
- Generated: Yes (by Python interpreter)
- Committed: No (gitignored globally)

**`.planning/` - GSD Planning**
- Purpose: Project planning documents
- Contains: Codebase analysis, phase plans
- Created by: GSD commands

**`runs/` - YOLO Training Runs**
- Purpose: Ultralytics training output
- Structure: Organized by YOLO automatically
- Key content: `runs/pose/arnis_stick_detector/`

---

*Structure analysis: 2026-04-18*
