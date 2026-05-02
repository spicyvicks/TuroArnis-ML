# Left Viewpoint Deployment Package

This directory contains models, templates, and inference code for **left-viewpoint** Arnis technique classification.

## Models

| Model | File | Template | Accuracy | Note |
|-------|------|----------|----------|------|
| v5 (mirrored) | `models/model_left_v5_mirrored.pth` | Mirrored | 88.70% | Best overall left accuracy |
| v6 (standard) | `models/model_left_v6_standard.pth` | Standard | 87.83% | Production architecture |

**Recommendation:** Use v6 for production (has `has_stick` flag + masked pooling). Use v5_mirrored if maximum accuracy is required.

## Templates

| File | Description | For Model |
|------|-------------|-----------|
| `src/feature_templates.json` | Standard left templates | v6_standard |
| `src/feature_templates_mirrored.json` | Mirrored left templates | v5_mirrored |

## Architecture Files

- `src/model_v5.py` — v5 HybridGCN (no masking, 46 hybrid features)
- `src/model_v6.py` — v6 HybridGCN (node masking, 49 hybrid features, has_stick flag)

## File Structure

```
left/
├── src/
│   ├── model_v5.py              # v5 architecture (with batch bugfix)
│   ├── model_v6.py              # v6 architecture (with batch bugfix)
│   ├── feature_templates.json   # Standard templates for left viewpoint
│   ├── feature_templates_mirrored.json  # Mirrored templates
│   └── inference_example.py     # Example usage script
├── models/
│   ├── model_left_v5_mirrored.pth   # Best v5 model (88.70%)
│   └── model_left_v6_standard.pth # Best v6 model (87.83%)
└── README.md
```

## Usage

See `src/inference_example.py` for a complete example.

Quick start:

```python
from deployment_package.left.src.model_v6 import load_deployment_model
from deployment_package.src.inference_v6 import V6Inference  # reuse front inference

# Point to left-specific model and templates
inf = V6Inference(
    model_path='deployment_package/left/models/model_left_v6_standard.pth',
    templates_path='deployment_package/left/src/feature_templates.json',
    stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
    device='cpu'
)

result = inf.predict(image, viewpoint='left')
```

## Important Notes

1. **Model architecture bugfix:** The `model_v5.py` and `model_v6.py` in this directory include a fix for batched inference (`hybrid_features.view(batch_size, -1)`). The original deployment_package/src/ files (for front viewpoint) do NOT have this fix yet and will fail on batch_size > 1.

2. **Left/right confusion:** From the left viewpoint, left-side and right-side techniques can be confused. Top confusion pairs from evaluation:
   - `left_elbow_block` ↔ `right_elbow_block`
   - `left_temple_block` ↔ `right_elbow_block`
   - `crown_thrust` ↔ `right_temple_block`

3. **Data size:** Left viewpoint has ~50% less training data than front (1,240 vs ~2,500 samples). Performance is expected to improve with more left-viewpoint recordings.

## Evaluation Summary (Real-Only Test, 115 samples)

| Model | Overall | Top-2 | Mean/Class |
|-------|---------|-------|------------|
| v5_mirrored | **88.70%** | 94.78% | **88.91%** |
| v6_standard | 87.83% | 94.78% | 84.71% |
| v5_standard | 85.22% | 94.78% | 82.49% |
| v6_mirrored | 82.61% | 93.91% | 78.89% |

Full results: `hybrid_classifier/models/left_viewpoint_real_only_evaluation.json`
