# Arnis Pose Classifier - Deployment Package

This package contains all necessary files to implement a standalone desktop application for real-time Arnis pose classification using the Hybrid GCN V2 specialist models.

## 📦 Package Contents

```
deployment_package/
├── models/                          # Trained GCN models
│   ├── hybrid_gcn_v2_front.pth     # Front viewpoint specialist (1.35 MB)
│   ├── hybrid_gcn_v2_left.pth      # Left viewpoint specialist (1.35 MB)
│   └── hybrid_gcn_v2_right.pth     # Right viewpoint specialist (1.35 MB)
│
├── weights/                         # YOLO stick detector
│   └── best.pt                      # YOLOv8-Pose stick detector (6.12 MB)
│
├── src/                             # Source code
│   ├── model_architecture.py        # HybridGCN model definition
│   ├── feature_extraction.py        # Feature extraction utilities
│   └── feature_templates.json       # Reference pose templates (202 KB)
│
├── docs/                            # Documentation
│   └── implementation_plan.md       # Comprehensive implementation guide
│
├── requirements.txt                 # Python dependencies (locked versions)
└── README.md                        # This file
```

## 🎯 Model Specifications

### Architecture: Hybrid GCN V2
- **Input**: 35 nodes (33 MediaPipe pose + 2 YOLO stick keypoints)
- **Node Features**: 6 per node (x, y, z, visibility, distance_to_hip, angle_from_hip)
- **Global Features**: 30 similarity scores (hybrid features)
- **Hidden Dimension**: 256
- **Layers**: 3 GCN layers + MLP fusion
- **Output**: 13 classes (Arnis stances)

### Classes
1. `crown_thrust_correct`
2. `left_chest_thrust_correct`
3. `left_elbow_block_correct`
4. `left_eye_thrust_correct`
5. `left_knee_block_correct`
6. `left_temple_block_correct`
7. `neutral_stance`
8. `right_chest_thrust_correct`
9. `right_elbow_block_correct`
10. `right_eye_thrust_correct`
11. `right_knee_block_correct`
12. `right_temple_block_correct`
13. `solar_plexus_thrust_correct`

## 🚀 Quick Start

### 1. Environment Setup

**Requirements**:
- Python 3.11
- Windows 10/11
- Intel Core 7 150U (1.80 GHz) or better
- 16 GB RAM
- No GPU required (CPU-optimized)

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` uses locked versions compatible with PyTorch 2.10.0 CPU. If you encounter issues, install PyTorch first:
> ```bash
> pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 2. Load Models

```python
import torch
from src.model_architecture import HybridGCN, CLASS_NAMES

# Load a specialist model (e.g., front viewpoint)
checkpoint = torch.load('models/hybrid_gcn_v2_front.pth', map_location='cpu')

model = HybridGCN(
    node_in_channels=checkpoint['node_feat_dim'],
    hybrid_in_channels=checkpoint['hybrid_feat_dim'],
    hidden_channels=checkpoint['hidden_dim'],
    num_classes=len(CLASS_NAMES),
    num_layers=checkpoint['num_layers'],
    dropout=checkpoint['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded! Test accuracy: {checkpoint['test_accuracy']:.2%}")
```

### 3. Extract Features from Image

```python
import json
from ultralytics import YOLO
from src.feature_extraction import extract_raw_features, extract_node_features, compute_hybrid_features

# Load YOLO stick detector
stick_detector = YOLO('weights/best.pt')

# Load feature templates
with open('src/feature_templates.json', 'r') as f:
    templates = json.load(f)

# Extract features from image
raw_data = extract_raw_features('path/to/image.jpg', stick_detector)

if raw_data:
    # Node features for GCN
    node_features = extract_node_features(
        raw_data['pose_keypoints'],
        raw_data['stick_keypoints']
    )
    
    # Hybrid features (similarity scores)
    hybrid_features = compute_hybrid_features(
        raw_data['global_features'],
        templates,
        viewpoint='front',  # or 'left', 'right'
        class_name='neutral_stance'  # target class for comparison
    )
    
    print(f"Node features shape: {node_features.shape}")  # (35, 6)
    print(f"Hybrid features shape: {hybrid_features.shape}")  # (30,)
```

### 4. Run Inference

```python
import torch
from torch_geometric.data import Data
from src.model_architecture import SKELETON_EDGES

# Create graph data
edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
graph = Data(
    x=torch.tensor(node_features, dtype=torch.float32),
    edge_index=edge_index,
    hybrid=torch.tensor(hybrid_features, dtype=torch.float32)
)

# Add batch dimension
batch = torch.zeros(35, dtype=torch.long)  # Single graph

# Inference
with torch.no_grad():
    output = model(graph.x, graph.edge_index, batch, graph.hybrid.unsqueeze(0))
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

print(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.2%})")
```

## 📊 Performance Expectations

### Inference Speed (Intel Core 7 150U, CPU-only)
- **MediaPipe Pose**: ~30ms per frame (33 FPS)
- **YOLO Stick Detector**: ~60ms per frame (17 FPS)
- **GCN Inference**: ~5ms per frame (200 FPS)
- **Full Pipeline**: ~95ms per frame (**~10 FPS**)

> ⚠️ **Performance Bottleneck**: The YOLO stick detector is the main bottleneck. See `docs/implementation_plan.md` for optimization strategies (quantization, frame skipping).

### Model Accuracy
- **Front Viewpoint**: Test accuracy varies by training run
- **Left Viewpoint**: Test accuracy varies by training run
- **Right Viewpoint**: Test accuracy varies by training run

Check the `test_accuracy` field in each `.pth` file for exact metrics.

## 🛠️ Next Steps

1. **Read the Implementation Plan**: See `docs/implementation_plan.md` for:
   - Complete desktop app architecture (PyQt6)
   - Database schema (SQLite)
   - Performance optimization strategies
   - Packaging instructions (PyInstaller)

2. **Answer Remaining Questions**: The implementation plan has 4 critical questions that need answers:
   - Database storage requirements
   - Viewpoint selection method
   - User experience flow
   - Packaging preference

3. **Build the Application**: Follow the implementation plan to create:
   - Real-time inference engine
   - Desktop GUI with camera feed
   - Database logging
   - Standalone executable

## 📝 Model Training Details

These models were trained using:
- **Training Script**: `4c_train_hybrid_gcn_v2.py`
- **Feature Generation**: `2b_generate_node_hybrid_features.py`
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: Random scaling (0.85-1.15) + jitter (2%)
- **Early Stopping**: Patience=20, overfitting threshold=20%

## 🔧 Troubleshooting

### Import Errors
If you get `ModuleNotFoundError` for `torch_geometric`:
```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

### YOLO Model Not Found
Ensure `weights/best.pt` exists. If missing, you need to train or obtain the stick detector model.

### MediaPipe Issues
If MediaPipe fails to initialize:
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.14
```

## 📄 License

This deployment package is for the TuroArnis-ML project. Ensure compliance with licenses for:
- PyTorch (BSD)
- MediaPipe (Apache 2.0)
- Ultralytics (AGPL-3.0)

## 📧 Support

For questions about implementation, refer to `docs/implementation_plan.md` or the original training repository.

---

**Package Version**: 1.0  
**Generated**: 2026-02-10  
**Compatible with**: PyTorch 2.10.0 CPU, Python 3.11
