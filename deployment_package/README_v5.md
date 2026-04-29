# V5 Front Model Deployment Package

**Status**: ✅ Ready for integration  
**Model**: `model_front_v5_deploy.pth`  
**Accuracy**: 77.7% real-only test (front view)  
**Classes**: 13 (including `neutral`)  
**Date**: 2026-04-28

---

## What's Included

| File | Purpose |
|------|---------|
| `models/model_front_v5_deploy.pth` | Trained model weights + config (861 KB) |
| `src/model_v5.py` | Model architecture (matches training exactly) |
| `src/feature_extraction_v5.py` | Feature extraction with signed features + v5 stick fallback |
| `src/inference_v5.py` | High-level inference wrapper for app integration |

---

## Model Specs

- **Architecture**: HybridGCN (GCN + hybrid MLP fusion)
- **Node features**: 6-dim `[x, y, z, vis, dist_to_hip, angle_from_hip]`
- **Hybrid features**: 46-dim (33 Gaussian similarity + 13 signed direction)
- **Hidden dim**: 128
- **Layers**: 3 GCN + 2-layer hybrid MLP
- **Parameters**: 69,541
- **Pooling**: `global_mean_pool` (no masking)
- **Classes**: 13 (12 techniques + neutral)

---

## Critical App Changes Required

### 1. Add `neutral` class back to app

The v5 model was trained with 13 classes including `neutral`. The app must support this:

```dart
// Flutter app class list
const List<String> CLASS_NAMES = [
  'crown_thrust_correct',
  'left_chest_thrust_correct',
  'left_elbow_block_correct',
  'left_eye_thrust_correct',
  'left_knee_block_correct',
  'left_temple_block_correct',
  'right_chest_thrust_correct',
  'right_elbow_block_correct',
  'right_eye_thrust_correct',
  'right_knee_block_correct',
  'right_temple_block_correct',
  'solar_plexus_thrust_correct',
  'neutral',  // <-- ADD THIS
];
```

### 2. Use v5 feature extraction

The app must compute **signed direction features** (13 additional features) that the v5 model expects:

```python
# In app feature extraction, add these 13 signed features:
'stick_tip_signed_x'
'grip_signed_x'
'wrist_spread'
'stick_reach'
'tip_height_vs_grip'
'stick_forearm_dot'
'tip_vs_nose_signed'
'tip_vs_shoulder_signed'
'left_elbow_angle_signed'
'right_elbow_angle_signed'
'stick_angle_signed'
'right_wrist_height_signed'
'left_wrist_height_signed'
```

See `feature_extraction_v5.py` for exact computation formulas.

### 3. Stick fallback behavior

When YOLO stick detection fails, use **origin-based fallback** (not NaN):

```python
stick_grip = [0.0, 0.0, 0.0, 0.0]  # origin fallback
stick_tip = [0.0, 0.0, 0.0, 0.0]
```

This creates a constant offset in `dist_to_hip`/`angle_from_hip` that the model learned to compensate for. **Do not use true zeros + masking** — that was v6 and performed worse.

---

## Quick Integration Example

```python
from deployment_package.src.inference_v5 import V5Inference

inf = V5Inference(
    model_path='deployment_package/models/model_front_v5_deploy.pth',
    templates_path='hybrid_classifier/feature_templates.json',
    stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
    device='cpu'
)

import cv2
image = cv2.imread('test.jpg')
result = inf.predict(image, viewpoint='front')

print(result['class'])        # 'crown_thrust_correct'
print(result['confidence'])   # 0.92
print(result['top_k'])        # [{'class': '...', 'confidence': ...}, ...]
```

---

## Per-Class Performance (Real-Only Test)

| Class | Accuracy |
|-------|----------|
| crown | 73.3% |
| left_chest | 33.3% |
| left_elbow | 20.0% |
| left_eye | ~80% |
| left_knee | ~80% |
| left_temple | ~85% |
| right_chest | ~90% |
| right_elbow | ~80% |
| right_eye | ~90% |
| right_knee | ~90% |
| right_temple | ~85% |
| solar_plexus | 100% |
| neutral | 100% |
| **Overall** | **77.7%** |

---

## Known Limitations

1. **Front-view only** — left/right viewpoint models not yet deployed
2. **CPU inference** — ~50-100ms per frame (10-20 FPS)
3. **Stick detection dependency** — YOLO required for best accuracy
4. **Left-side classes weaker** — crown (73%), left_chest (33%), left_elbow (20%)

---

## Next Steps

1. [ ] Integrate `inference_v5.py` into TuroArnis app
2. [ ] Add `neutral` class to app UI/class list
3. [ ] Implement signed feature computation in app feature extraction
4. [ ] Test end-to-end with real camera feed
5. [ ] Train and deploy left/right viewpoint models

---

## Files to Replace in App

| Old File | New File |
|----------|----------|
| `app/models/gcn/model_architecture.py` | `deployment_package/src/model_v5.py` |
| `app/models/gcn/feature_extraction.py` | `deployment_package/src/feature_extraction_v5.py` |
| `app/models/gcn/inference.py` | `deployment_package/src/inference_v5.py` |
| `models/active_model.json` | Already updated to point to v5 |

---

## Dependencies

Same as existing deployment:
- torch >= 2.0
- torch-geometric >= 2.3
- ultralytics >= 8.0
- mediapipe >= 0.10
- opencv-python >= 4.8
