---
updated_at: "2026-04-22T13:42:40Z"
---

## Architecture Overview

TuroArnis-ML is a **hybrid Graph Convolutional Network (GCN) classifier** for martial arts (Arnis) technique recognition from pose keypoints. The architecture combines **node-level spatial features** (body + stick keypoints) with **global hybrid features** (similarity scores to reference poses).

## Key Components

| Component | Path | Responsibility |
|-----------|------|---------------|
| Training Script | `hybrid_classifier/4c_train_hybrid_gcn_v2.py` | Train per-viewpoint or merged HybridGCN models |
| Feature Generator | `hybrid_classifier/2b_generate_node_hybrid_features.py` | Extract node + hybrid features from images |
| Model Architecture | `deployment_package/src/model_architecture.py` | Inference-ready HybridGCN model class |
| Feature Extraction | `deployment_package/src/feature_extraction.py` | Real-time feature extraction for inference |
| Templates | `hybrid_classifier/feature_templates.json` | Reference pose statistics per class/viewpoint |
| Deployed Models | `deployment_package/models/*.pth` | Trained model weights (front/left/right) |

## Data Flow

```
Image → MediaPipe Pose → 33 Body Keypoints (x,y,z,vis)
     ↓
     → YOLO Stick Detector → 2 Stick Keypoints (grip, tip)
     ↓
     → Method 4 Correction (pinky snap + shin-based length)
     ↓
     → 35 Nodes (33 pose + 2 stick)
     ↓
     → Node Features [35, 6]: [x, y, z, vis, dist_to_hip_3d, angle_from_hip]
     ↓
     → Hybrid Features [30]: Gaussian similarity to reference templates
     ↓
     → HybridGCN (3-layer GCN + MLP fusion) → 12/13 Class Logits
```

## Model Architecture

**HybridGCN V2** (`deployment_package/src/model_architecture.py`):

1. **Node Embedding**: 35 nodes × 8-dim learnable embeddings
2. **GCN Layers**: 3 × GCNConv with BatchNorm + ReLU + Dropout
3. **Global Pooling**: Mean pool node features → graph representation
4. **Hybrid MLP**: 2-layer MLP processes 30 hybrid features
5. **Fusion**: Concatenate GCN output + hybrid features
6. **Classification**: Linear layer → class logits

**Hyperparameters (Feb 13, 2026 80% models):**
- Hidden dim: 128 (front), 256 (left/right)
- Node embed dim: 8
- Dropout: 0.5
- Learning rate: 0.001
- Classes: 13 (including neutral_stance)

## Conventions

- **Class names**: `{position}_{technique}_{variant}` (e.g., `front_left_chest_thrust_correct`)
- **Skeleton edges**: Bidirectional edges connecting body joints + stick (28 edges)
- **Viewpoints**: front, left, right (trained separately)
- **Feature templates**: JSON with per-class/viewpoint mean/std statistics
- **Normalization**: Features normalized by hip center

## Critical Issues Identified

1. **NODE_EMBED_DIM mismatch**: Current code uses 16, models trained with 8
2. **Class count mismatch**: Deployment models have 12 classes, code expects 13
3. **Class name mismatch**: `neutral_stance` vs `neutral`
4. **Hidden dim inconsistency**: Left/right models use 256, front uses 128

See `INVESTIGATION_REPORT_MARCH2_MODELS.md` for detailed analysis.
