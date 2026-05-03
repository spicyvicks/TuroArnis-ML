# Right Viewpoint Deployment Package

This directory contains models, templates, and inference code for **right-viewpoint** Arnis technique classification.

## Models

| Model | File | Template | Accuracy | Note |
|-------|------|----------|----------|------|
| v5 (standard) | `models/model_right_v5_standard.pth` | Standard | **69.15%** | Best overall right accuracy |
| v6 (standard) | `models/model_right_v6_standard.pth` | Standard | 67.02% | Production architecture |

**Recommendation:** Use v5 for right viewpoint (69.15% > 67.02%). v5 uses standard templates and 46 hybrid features.

## Templates

| File | Description | For Model |
|------|-------------|-----------|
| `src/feature_templates.json` | Standard templates for all viewpoints | v5_standard, v6_standard |
| `src/feature_templates_mirrored.json` | Mirrored templates (for future use) | — |

## Architecture Files

- `src/model_v5.py` — v5 HybridGCN (no masking, 46 hybrid features)
- `src/model_v6.py` — v6 HybridGCN (node masking, 49 hybrid features, has_stick flag)

## File Structure

```
right/
├── src/
│   ├── model_v5.py              # v5 architecture (with batch bugfix)
│   ├── model_v6.py              # v6 architecture (with batch bugfix)
│   ├── feature_templates.json   # Standard templates for right viewpoint
│   ├── feature_templates_mirrored.json  # Mirrored templates (unused)
│   └── inference_example.py     # Example usage script
├── models/
│   ├── model_right_v5_standard.pth   # Best v5 model (69.15%)
│   └── model_right_v6_standard.pth # v6 model (67.02%)
└── README.md
```

## Usage

See `src/inference_example.py` for a complete example.

Quick start:

```python
from deployment_package.right.src.model_v5 import load_deployment_model
from deployment_package.src.inference_v5 import V5Inference  # reuse front inference

# Point to right-specific model and templates
inf = V5Inference(
    model_path='deployment_package/right/models/model_right_v5_standard.pth',
    templates_path='deployment_package/right/src/feature_templates.json',
    stick_model_path='runs/pose/stick_detector_20260425_212025/weights/best.pt',
    device='cpu'
)

result = inf.predict(image, viewpoint='right')
```

## Important Notes

1. **Model architecture bugfix:** The `model_v5.py` and `model_v6.py` in this directory include a fix for batched inference (`hybrid_features.view(batch_size, -1)`). The original `deployment_package/src/` files (for front viewpoint) now also have this fix applied.

2. **Left/right confusion:** From the right viewpoint, left-side and right-side techniques can be confused. Top confusion pairs from evaluation:
   - `left_temple_block` -> `right_temple_block` (7 samples)
   - `left_elbow_block` -> `right_elbow_block` (5 samples)
   - `solar_plexus_thrust` -> `right_elbow_block` (3 samples)

3. **Data size:** Right viewpoint has ~2,100 training samples with ~94 test samples. Performance is expected to improve with more right-viewpoint recordings.

4. **Accuracy gap vs front:** Right viewpoint (69.15%) is below front viewpoint (77.7%). This is expected because right-viewpoint data has more challenging left/right symmetry confusion.

## Training Data Experiments (Verified 2026-05-03)

We ran two experiments to test whether horizontal-flip augmentation in the right-viewpoint training data was harmful noise or useful diversity.

### Experiment 1: Cleaned Dataset (Removed All Flipped Augmentations)
- Removed all `*aug2*` and `*aug3*` flipped images from training set (1105 unflipped images remaining)
- Trained v5 (3x/5x synthetic) and v6 (2x/5x synthetic) on cleaned data
- **Result**: All models crashed to **31–45% accuracy** (down from 69%)

### Experiment 2: 0-Flip Re-Augmentation (More Data, No Flips)
- Deleted all augmented files and regenerated 2208 training images (4 per original, **0% flip rate**)
- Trained v5 (standard + mirrored templates) and v6 (standard + mirrored templates)
- **Result**: All models crashed to **33–45% accuracy**

### Conclusion

**Horizontal flips are essential for the right-viewpoint model, not noise.** The model requires seeing both the original pose and its mirror to learn left-vs-right technique distinctions from a right-side camera. Removing flips — even with more data volume — destroys accuracy by ~24–36 percentage points.

The current deployed model (`model_right_v5_standard.pth`) was trained on the original mixed data (with flips) and achieves **69.15%** real-only accuracy. This remains the **definitive right-viewpoint champion**.

## Evaluation Summary (Real-Only Test, 94 samples)

### Deployed Models

| Model | Overall | Top-2 | Mean/Class | Status |
|-------|---------|-------|------------|--------|
| **v5_standard (deployed)** | **69.15%** | **85.11%** | **67.70%** | **Champion** |
| v6_standard (deployed) | 67.02% | 85.11% | 63.48% | Alternative |

### Failed Experiments (for reference)

| Model | Overall | Note |
|-------|---------|------|
| v5_cleaned_3x | 42.55% | Removed flips, 3x synthetic |
| v5_cleaned_5x | 44.68% | Removed flips, 5x synthetic |
| v6_cleaned_2x | 31.91% | Removed flips, 2x synthetic |
| v6_cleaned_5x | 34.04% | Removed flips, 5x synthetic |
| v5_reaugmented (0 flip) | 45.2% | 2208 samples, 0 flip, 3x synth |
| v5_mirrored_reaugmented (0 flip) | 43.6% | 2208 samples, 0 flip, mirrored templates |
| v6_reaugmented (0 flip) | 33.0% | 2208 samples, 0 flip, 2x synth |
| v6_mirrored_reaugmented (0 flip) | 34.0% | 2208 samples, 0 flip, mirrored templates |

Full results: `hybrid_classifier/models/right_viewpoint_real_only_evaluation.json`
