# V6 Front Model Deployment Package

**Status**: ✅ Ready for integration  
**Model**: `model_front_v6_deploy.pth`  
**Accuracy**: 74.2% real-only test (front view)  
**Classes**: 13 (including `neutral`)  
**Date**: 2026-04-30  
**Source**: `model_front_with_synthetic_2x_v6.pth` (epoch 61, val_acc 74.7%)

---

## What's Included

| File | Purpose |
|------|---------|
| `models/model_front_v6_deploy.pth` | Trained model weights + config (13 classes) |
| `src/model_v6.py` | V6 architecture with **masked pooling** + `node_mask` |
| `src/feature_extraction_v6.py` | True zero-stick fallback + **7-dim node features** + `node_mask` |
| `src/inference_v6.py` | High-level `V6Inference` class |

---

## Model Specs

- **Architecture**: HybridGCN with masked global mean pool
- **Node features**: **7-dim** `[x, y, z, vis, dist_to_hip, angle_from_hip, has_stick]`
- **Hybrid features**: 49-dim (33 Gaussian + 15 signed + 1 has_stick binary)
- **Hidden dim**: 128
- **Layers**: 3 GCN + 2-layer hybrid MLP
- **Parameters**: 69,861
- **Pooling**: `global_add_pool` with `node_mask` division (masked mean)
- **Classes**: 13 (12 techniques + neutral)

---

## Key Differences from V5

| Aspect | V5 | V6 |
|--------|-----|-----|
| **Accuracy** | **76.3%** | 72.2% |
| **Node features** | 6-dim | **7-dim** (+ `has_stick` binary) |
| **Hybrid features** | 46 | 49 (+ `has_stick` binary) |
| **Stick fallback** | Origin-based offset | **True zeros** `[0,0,0,0]` |
| **Pooling** | `global_mean_pool` (all 35 nodes) | **Masked** — zero-stick nodes excluded |
| **Architecture** | Standard | **Node masking** in forward pass |
| **Philosophy** | "Bug as feature" — origin offset is learnable signal | **Correct** — missing nodes should not contribute |

---

## Critical App Changes Required

### 1. `node_mask` is REQUIRED

The V6 model **will crash** without `node_mask` in the PyG Data object:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,      # [35, 7]
    edge_index=edge_index,
    hybrid_features=hybrid_features,  # [49]
    y=label,
    node_mask=node_mask    # [35] — REQUIRED for v6, optional for v5
)
```

`node_mask` generation:
```python
def create_node_mask(has_stick_detected):
    mask = np.ones(35, dtype=np.float32)
    if not has_stick_detected:
        mask[33:] = 0.0  # mask out stick nodes
    return mask
```

### 2. 7-dim node features

V6 expects **7 dimensions** per node, not 6:
```python
# V6 node feature: [x, y, z, vis, dist_to_hip, angle_from_hip, has_stick]
#   has_stick = 1.0 for body nodes (0-32)
#   has_stick = 1.0 for stick nodes (33-34) if YOLO detected
#   has_stick = 0.0 for stick nodes (33-34) if zero-stick fallback
```

### 3. True zero-stick fallback

When YOLO fails, use **true zeros** (not origin-based offset like v5):
```python
stick_grip = [0.0, 0.0, 0.0, 0.0]
stick_tip = [0.0, 0.0, 0.0, 0.0]
```

### 4. 49 hybrid features

V6 expects **49 hybrid features** (v5 expects 46). The additional 3 are:
- `left_wrist_x_signed`
- `right_wrist_x_signed`
- `has_stick` (binary hybrid feature)

### 5. Add `neutral` class

Same as v5 — app must support 13 classes including `neutral`.

---

## Quick Integration Example

```python
from deployment_package.src.inference_v6 import V6Inference

inf = V6Inference(
    model_path='deployment_package/models/model_front_v6_deploy.pth',
    templates_path='hybrid_classifier/feature_templates.json',
    stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
    device='cpu'
)

import cv2
image = cv2.imread('test.jpg')
result = inf.predict(image, viewpoint='front')

print(result['class'])        # 'crown_thrust_correct'
print(result['confidence'])   # 0.88
print(result['top_k'])        # [{'class': '...', 'confidence': ...}, ...]
```

---

## Per-Class Performance (Real-Only Test)

| Class | Accuracy |
|-------|----------|
| crown | 66.7% |
| left_chest | 46.7% |
| left_elbow | 7.1% |
| left_eye | 100.0% |
| left_knee | 71.4% |
| left_temple | 84.2% |
| right_chest | 69.2% |
| right_elbow | 58.3% |
| right_eye | 84.6% |
| right_knee | 77.8% |
| right_temple | 93.3% |
| solar_plexus | 85.7% |
| neutral | 100.0% |
| **Overall** | **74.2%** |

---

## Why Deploy V6 Instead of V5?

| Advantage | Detail |
|-----------|--------|
| **Architecturally correct** | Missing nodes are properly excluded from pooling |
| **No origin offset hack** | V5's 77.7% relies on a data artifact that may break with different image sizes |
| **Generalizable masking** | `node_mask` mechanism works for any viewpoint with missing detections |
| **Debuggable** | `has_stick` node feature tells the GCN which nodes are real vs missing during message passing |
| **Future-proof** | Masking is the standard approach in GNN literature |

**Trade-off**: 4.1% lower accuracy (72.2% vs 76.3%) but cleaner, more robust architecture.

---

## Files to Replace in App

| Old File | New File |
|----------|----------|
| `app/models/gcn/model_architecture.py` | `deployment_package/src/model_v6.py` |
| `app/models/gcn/feature_extraction.py` | `deployment_package/src/feature_extraction_v6.py` |
| `app/models/gcn/inference.py` | `deployment_package/src/inference_v6.py` |

**⚠️ CRITICAL**: App must pass `node_mask` into PyG Data object. V6 forward pass crashes without it.

---

## Dependencies

Same as v5:
- torch >= 2.0
- torch-geometric >= 2.3
- ultralytics >= 8.0
- mediapipe >= 0.10
- opencv-python >= 4.8
