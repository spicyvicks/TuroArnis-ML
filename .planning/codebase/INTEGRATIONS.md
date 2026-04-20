# External Integrations

**Analysis Date:** 2026-04-18

## Computer Vision Services

**MediaPipe (Google):**
- **Package:** `mediapipe==0.10.14`
- **Usage:** Human pose estimation (33 keypoints)
- **Implementation:** `mp.solutions.pose.Pose`
- **Model Complexity:** 2 (highest accuracy mode)
- **Files:** `inference_realtime_gcn.py`, `feature_extraction.py`, all feature extraction scripts
- **Key API:**
  ```python
  mp_pose.Pose(
      static_image_mode=False,
      model_complexity=2,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5
  )
  ```

**Ultralytics YOLO:**
- **Package:** `ultralytics==8.1.29`
- **Usage:** Stick detection with pose keypoints
- **Implementation:** `YOLO(model_path)`
- **Model:** Custom-trained on Arnis stick dataset
- **Files:** All scripts requiring stick detection
- **Key API:**
  ```python
  YOLO('runs/pose/arnis_stick_detector/weights/best.pt')
  ```

## ML Framework Ecosystem

**PyTorch Hub:**
- Source for pre-trained YOLO models
- Custom models loaded from local paths

**PyTorch Geometric (PyG):**
- **Installation:** Requires special wheel index for CUDA compatibility
- **URL:** `https://data.pyg.org/whl/torch-2.2.1+cu118.html`
- **Components Used:**
  - `GCNConv` - Graph convolution layers
  - `global_mean_pool` - Graph-level pooling
  - `Data`, `Batch` - Graph data structures
  - `DataLoader` - Batch loading for graphs

## Data Storage

**Local Filesystem (Primary):**
- All data stored locally, no cloud database
- Images: JPEG format in `dataset/`, `dataset_split/`
- Model checkpoints: PyTorch `.pth` files
- Features: PyTorch `.pt` tensors
- Templates: JSON files

**Key Data Locations:**
| Location | Format | Purpose |
|----------|--------|---------|
| `dataset/` | JPG/PNG | Raw training images |
| `dataset_split/train/` | JPG/PNG | Training split |
| `dataset_split/test/` | JPG/PNG | Test split |
| `dataset_graphs/` | PyTorch Geometric | Pre-computed graph datasets |
| `hybrid_classifier/hybrid_features_v3/` | .pt | Feature tensors |
| `reference_poses/` | JPG | Reference images per class/viewpoint |

**Configuration/Feature Storage:**
- `hybrid_classifier/feature_templates.json` - Feature statistics per class
- `hybrid_classifier/feature_templates_mirrored.json` - Mirrored variant
- `models/active_model.json` - Active model registry
- `models/model_registry.json` - Full model registry

## Pre-trained Models

**YOLO Models (Downloaded):**
- `yolov8n-pose.pt` - Pre-trained pose estimation (COCO)
- `yolov8n.pt` - Base detection model
- `runs/pose/arnis_stick_detector/weights/best.pt` - Custom stick detector

**GCN Checkpoints:**
- `models/gcn_checkpoints/best_model.pth` - Best performing GCN
- `models/geopose_model.pt` - Alternative pose-based model
- `hybrid_classifier/models/hybrid_gcn_v2.pth` - Hybrid model

## Hardware Interfaces

**Webcam (OpenCV):**
- **Usage:** Real-time inference
- **Implementation:** `cv2.VideoCapture(camera_id)`
- **Files:** `inference_realtime_gcn.py`
- **Default Camera:** 0 (primary webcam)

**GPU Acceleration (Optional):**
- **CUDA Version:** 11.8 (development)
- **CPU Fallback:** Automatic when GPU unavailable
- **Inference Target:** CPU-optimized for deployment

## Environment Configuration

**No External Secrets:**
- No `.env` files in repository
- No API keys required
- All models run locally

**Command-line Arguments:**
All scripts use `argparse` for configuration:
```python
parser.add_argument('--model', type=str, default='models/gcn_checkpoints/best_model.pth')
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--viewpoint', choices=['front', 'left', 'right'])
```

## Integration Architecture

**Multi-Model Pipeline:**
```
Webcam Frame
    ↓
MediaPipe Pose → 33 Body Keypoints
    ↓
YOLO Stick Detector → 2 Stick Keypoints (grip + tip)
    ↓
Graph Construction → 35 Nodes (33 + 2)
    ↓
Spatial GCN → 13 Class Probabilities
    ↓
Temporal Smoothing (deque buffer) → Stable Prediction
    ↓
OpenCV Display → Annotated Frame
```

**Hybrid Classifier Pipeline:**
```
Image
    ↓
MediaPipe + YOLO → Pose + Stick Features
    ↓
Geometric Feature Extraction → 30+ Measurements
    ↓
Gaussian Similarity vs Templates → Similarity Scores
    ↓
Random Forest / MLP → Class Prediction
```

## Deployment Integration

**PyInstaller Build:**
- Target: Single Windows executable
- Includes: Python runtime, PyTorch, all dependencies
- GUI: PyQt6 interface for deployment

**No Cloud Services:**
- No AWS, GCP, Azure dependencies
- No REST APIs
- No authentication services
- Fully offline-capable

---

*Integration audit: 2026-04-18*
