# Deployment Package Manifest
# Updated: 2026-05-03

## Package Information
- **Version**: v5_right_confirmed
- **Target Platform**: Windows Desktop (CPU-only)
- **Python Version**: 3.11
- **PyTorch Version**: 2.10.0+cpu

## Active Models by Viewpoint

### Front Viewpoint
**`front/models/model_front_v5_standard.pth`** — Front-view specialist, 74.7% real-only accuracy
- 13 classes (including `neutral`)
- 6-dim node features + 46 hybrid features
- CPU inference optimized
- Source: `model_front_with_synthetic_3x_v5.pth` (epoch 22, val_acc 58.8%)

**`front/models/model_front_v6_standard.pth`** — Front v6 production model, 74.2% real-only accuracy
- 7-dim node features + 49 hybrid features (has_stick flag)
- Masked pooling for zero-stick fallback
- More robust architecture

### Left Viewpoint
**`left/models/model_left_v5_mirrored.pth`** — Left-view specialist, 88.70% real-only accuracy
- Mirrored templates (selfie-style camera)
- Best accuracy for left viewpoint

**`left/models/model_left_v6_standard.pth`** — Left v6 production model, 87.83% real-only accuracy
- Standard templates
- 7-dim node features with masked pooling

### Right Viewpoint
**`right/models/model_right_v5_standard.pth`** — Right-view specialist, 69.15% real-only accuracy
- Standard templates
- Best accuracy for right viewpoint
- 6-dim node features + 46 hybrid features

**`right/models/model_right_v6_standard.pth`** — Right v6 alternative model, 67.02% real-only accuracy
- Standard templates
- 7-dim node features with masked pooling

## Package Structure (Viewpoint-Organized)

### `front/` — Front Viewpoint Deployment
- `models/` — `model_front_v5_standard.pth`, `model_front_v6_standard.pth`
- `src/` — `model_v5.py`, `model_v6.py` (with batch bugfix + smart loading)
- `src/` — `feature_templates.json`, `feature_templates_mirrored.json`
- `README.md` — Front viewpoint documentation

### `left/` — Left Viewpoint Deployment
- `models/` — `model_left_v5_mirrored.pth` (88.70%), `model_left_v6_standard.pth` (87.83%)
- `src/` — `model_v5.py`, `model_v6.py` (with batch bugfix + smart loading)
- `src/` — `feature_templates.json`, `feature_templates_mirrored.json`
- `README.md` — Left viewpoint documentation

### `right/` — Right Viewpoint Deployment
- `models/` — `model_right_v5_standard.pth` (69.15%), `model_right_v6_standard.pth` (67.02%)
- `src/` — `model_v5.py`, `model_v6.py` (with batch bugfix + smart loading)
- `src/` — `feature_templates.json`, `feature_templates_mirrored.json`
- `README.md` — Right viewpoint documentation

### Legacy Files (Backward Compatibility)
Original files remain in root `deployment_package/`:
- `src/model_v5.py` — With batch bugfix (updated 2026-05-02)
- `src/model_v6.py` — With batch bugfix (updated 2026-05-02)
- `models/model_front_v5_deploy.pth` — Same as `front/models/model_front_v5_standard.pth`
- `models/model_front_v6_deploy.pth` — Same as `front/models/model_front_v6_standard.pth`
- `src/feature_templates.json` — Same as `front/src/feature_templates.json`

### Weights
- `weights/best.pt` (6.12 MB) — YOLOv8-Pose stick detector

### Feature Extraction
- `src/feature_extraction_v5.py` — V5 feature extraction with signed features
- `src/feature_extraction_v6.py` — V6 feature extraction with 7-dim nodes + node_mask
- `src/feature_extraction.py` — Legacy extraction (superseded)

### Inference Wrappers
- `src/inference_v5.py` — High-level inference wrapper for app integration
- `src/inference_v6.py` — V6 high-level inference wrapper

### Documentation
- `README_v5.md` — V5 deployment guide
- `README_v6.md` — V6 deployment guide
- `docs/implementation_plan.md` — Legacy comprehensive guide
- `README.md` — Legacy quick start

### Configuration
- `requirements.txt` — Python dependencies (locked versions)

## Total Package Size
Approximately **10 MB** (excluding Python environment)

## Deployment Checklist

### Pre-Implementation
- [x] Front model trained and validated (74.7% real-only)
- [x] Left model trained and validated (88.70% real-only)
- [x] Right model trained and validated (69.15% real-only)
- [x] Deployment checkpoints created with config
- [x] Feature extraction verified against training
- [ ] **App update: Add `neutral` class to class list (13 classes)**
- [ ] **App update: Implement signed direction features in feature extraction**
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies from `requirements.txt`

### Implementation Phase
- [ ] Integrate `inference_v5.py` / `inference_v6.py` into TuroArnis app
- [ ] Test end-to-end with real camera feed
- [x] Viewpoint switching (front/left/right) — all 3 models ready

## Known Limitations

1. ~~Right viewpoint pending~~ — **COMPLETED** 2026-05-02: `right/` deployment package created
2. **Performance**: ~10-20 FPS on Intel Core 7 150U
   - **Solution**: Frame skipping or YOLO quantization
3. **CPU-Only**: No GPU acceleration
4. **Batch bugfix applied to all packages** — `front/`, `left/`, and `right/` model files include the fix. Root `src/` files also updated 2026-05-02.
5. **Left-side classes weaker in front models**: crown 67%, left_chest 40%, left_elbow 7% (front viewpoint only)

## Right Viewpoint Dataset Cleaning Experiment (2026-05-02)

**Hypothesis**: Horizontal-flip augmentation (`aug2`, `aug3`) corrupts right-viewpoint training by mirroring left/right body geometry while keeping original labels, causing systematic left/right confusion.

**Method**: Removed all `*aug2*` and `*aug3*` flipped images from training set (1105 unflipped images remaining, down from ~2163). Trained v5 (3x/5x synthetic) and v6 (2x/5x synthetic) models on cleaned data.

**Results** (tested on same 94-sample real test set):
- `v5_standard` (original, with flipped): **69.15%** overall | 85.11% top-2 | 67.7% mean/class
- `v5_cleaned_3x`: 42.55% overall | 53.19% top-2 | 36.9% mean/class
- `v5_cleaned_5x`: 44.68% overall | 58.51% top-2 | 40.4% mean/class
- `v6_cleaned_2x`: 31.91% overall | 54.26% top-2 | 25.9% mean/class
- `v6_cleaned_5x`: 34.04% overall | 47.87% top-2 | 26.4% mean/class

**Conclusion**: Removing flipped data degraded accuracy by **~25 percentage points**. The data volume reduction (1080 vs 2163 samples) outweighed any label-noise benefit. **Original `v5_standard` model remains the right-viewpoint champion.** Flipped augmentations, despite introducing left/right ambiguity, improve generalization for the right-viewpoint specialist.

## Right Viewpoint 0-Flip Re-Augmentation Experiment (2026-05-03)

**Hypothesis**: If data volume loss was the cause of the cleaning failure, re-augmenting with **zero horizontal flips** but **more copies per image** (4 total: 1 original + 3 non-flip augmented) should recover accuracy.

**Method**: Deleted all 1611 existing augmented files from `dataset_split/train/right/`, then generated 2208 new training images (4 per original, 0% flip rate). Extracted v5/v6 features with standard and mirrored templates, generated synthetic data, and trained 4 models.

**Results** (tested on same 94-sample real test set):
- `v5_standard reaugmented` (0 flip, 3x synth): **45.2%** overall
- `v5_mirrored reaugmented` (0 flip, 3x synth): **43.6%** overall
- `v6_standard reaugmented` (0 flip, 2x synth): **33.0%** overall
- `v6_mirrored reaugmented` (0 flip, 2x synth): **34.0%** overall
- `v5_standard` (original, **with** flipped aug): **69.15%** overall

**Conclusion**: The 0-flip re-augmentation experiment **failed catastrophically** (33–45% vs 69%). Horizontal flips are **essential** for the right-viewpoint model, not noise. The model requires seeing both the original pose and its mirror to learn left-vs-right technique distinctions from a right-side camera. **The `v5_standard` model at 69.15% remains the definitive right-viewpoint champion.**

## Version History

### v6_front (2026-04-30)
- **V6 front specialist model redeployed** (from retrained model)
- 74.2% real-only test accuracy (masked pooling architecture)
- 13 classes with `neutral` support
- 7-dim node features (has_stick binary on nodes)
- 49 hybrid features (signed + has_stick)
- True zero-stick fallback with node_mask masking
- Source: `model_front_with_synthetic_2x_v6.pth` (epoch 61, val_acc 74.7%)
- Cleaner architecture, 0.5% below v5 but more robust

### v6_front (2026-04-28)
- **V6 front specialist model deployed**
- 74.6% real-only test accuracy (masked pooling architecture)
- 13 classes with `neutral` support
- 7-dim node features (has_stick binary on nodes)
- 49 hybrid features (signed + has_stick)
- True zero-stick fallback with node_mask masking
- Cleaner architecture, 3.1% below v5 but more robust

### v5_front (2026-04-30)
- **V5 front specialist model redeployed** (from retrained model)
- 74.7% real-only test accuracy (exceeds 70% target)
- 13 classes with `neutral` support
- Signed direction features (13 hybrid features)
- V5 stick fallback (origin-based)
- Source: `model_front_with_synthetic_3x_v5.pth` (epoch 22, val_acc 58.8%)
- **RECOMMENDED for production** — higher accuracy

### v5_front (2026-04-28)
- **V5 front specialist model deployed**
- 77.7% real-only test accuracy (exceeds 70% target)
- 13 classes with `neutral` support
- Signed direction features (13 hybrid features)
- V5 stick fallback (origin-based, proven 77.7%)
- **RECOMMENDED for production** — higher accuracy

### v1.1 (2026-02-11)
- Updated Class List (Removed 'neutral_stance', 13 -> 12 classes)
- Syncing latest Hybrid GCN V2 models
- Updated feature extraction logic

### v1.0 (2026-02-10)
- Initial deployment package
- 3 specialist models (Front/Left/Right)
- YOLO stick detector
- Feature extraction utilities
- Comprehensive documentation
