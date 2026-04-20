# Technology Stack

**Analysis Date:** 2026-04-18

## Languages

**Primary:**
- **Python 3.12** - All ML pipeline code, model training, and inference

**Secondary:**
- **JSON** - Configuration files and feature templates
- **Markdown** - Documentation and guides
- **YAML** - YOLO training configuration (`arnis_stick_detector/args.yaml`)

## Runtime

**Environment:**
- Python 3.12 (indicated by PyTorch compatibility)
- Both CPU and CUDA 11.8 GPU support

**Package Managers:**
- **pip** - Python package installation
- Two requirements files for different environments:
  - `requirements_gcn.txt` - Development with CUDA 11.8
  - `deployment_package/requirements.txt` - Production CPU-only deployment

## Core Frameworks

**Machine Learning & Deep Learning:**

| Framework | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.2.1 (dev) / 2.10.0 (deploy) | Core deep learning framework |
| PyTorch Geometric | 2.5.2 (dev) / 2.4.0 (deploy) | Graph Neural Network operations |
| torch-scatter | 2.1.2 | Scatter operations for PyG |
| torch-sparse | 0.6.18 | Sparse tensor operations |
| torch-cluster | 1.6.3 | Clustering algorithms for graphs |
| torch-spline-conv | 1.2.2 | Spline-based convolutions |

**Computer Vision:**

| Framework | Version | Purpose |
|-----------|---------|---------|
| Ultralytics YOLO | 8.1.29 (dev) / 8.3.252 (deploy) | Stick detection with pose estimation |
| MediaPipe | 0.10.14 | Body pose keypoint extraction |
| OpenCV | 4.9.0.80 (dev) / 4.13.0.92 (deploy) | Image processing and video capture |
| Pillow | 10.2.0 | Image manipulation |

**Traditional ML & Data Science:**

| Framework | Version | Purpose |
|-----------|---------|---------|
| scikit-learn | 1.4.1.post1 | Random Forest, XGBoost, metrics |
| NumPy | 1.26.4 | Numerical computations |
| Pandas | 2.2.1 | Data manipulation |
| SciPy | 1.17.0 | Scientific computing |

**Data Augmentation:**
- **Albumentations** 1.4.0 - Image augmentation pipeline

## GUI & Deployment

| Tool | Version | Purpose |
|------|---------|---------|
| PyQt6 | 6.8.0 | Desktop application GUI |
| PyInstaller | 6.12.0 | Executable packaging |

## Visualization

| Tool | Version | Purpose |
|------|---------|---------|
| Matplotlib | 3.8.3 | Static plotting |
| Seaborn | 0.13.2 | Statistical visualization |

## Key Dependencies by Component

**Graph Neural Network (GCN) Pipeline:**
- `torch_geometric` - Graph convolution layers
- `torch_geometric.nn` - GCNConv, GATConv, global_mean_pool
- `torch_geometric.data` - Data, Batch for graph construction

**Real-time Inference (`inference_realtime_gcn.py`):**
- OpenCV - Webcam capture and display
- MediaPipe - Real-time pose detection
- Ultralytics YOLO - Stick detection
- PyTorch Geometric - GCN model inference

**Hybrid Classifier (`hybrid_classifier/`):**
- scikit-learn - Random Forest, MLPClassifier
- joblib - Model serialization

**Deployment Package (`deployment_package/`):**
- CPU-optimized PyTorch 2.10.0+cpu
- PyQt6 - GUI application
- PyInstaller - Build standalone executable

## Configuration Files

**Environment Configuration:**
- No `.env` files detected - configuration via command-line arguments
- Model paths hardcoded with sensible defaults
- JSON-based feature templates store reference pose statistics

**Build/Development:**
- `.gitignore` - Excludes large model files and datasets
- `requirements_gcn.txt` - Development dependencies with CUDA
- `deployment_package/requirements.txt` - Production dependencies

## Platform Requirements

**Development:**
- Python 3.12
- CUDA 11.8 (optional, for GPU training)
- ~8GB+ RAM for dataset processing
- Webcam for real-time testing

**Production/Deployment:**
- Windows 10/11 (PyInstaller build target)
- CPU-only PyTorch build
- No GPU required for inference

**Model Files (Binary Assets):**
- `yolov8n-pose.pt` (6.8 MB) - YOLO pose model for stick detection
- `yolov8n.pt` (6.5 MB) - YOLO base model
- `models/gcn_checkpoints/best_model.pth` - Trained GCN weights
- `models/geopose_model.pt` - Alternative pose model

---

*Stack analysis: 2026-04-18*
