# Deployment Package Manifest
# Updated: 2026-04-30

## Package Information
- **Version**: v5_front
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

### Legacy Files (Backward Compatibility)
Original files remain in root `deployment_package/`:
- `src/model_v5.py` — Original (no batch bugfix)
- `src/model_v6.py` — Original (no batch bugfix)
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
- [x] Deployment checkpoints created with config
- [x] Feature extraction verified against training
- [ ] **App update: Add `neutral` class to class list (13 classes)**
- [ ] **App update: Implement signed direction features in feature extraction**
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies from `requirements.txt`

### Implementation Phase
- [ ] Integrate `inference_v5.py` / `inference_v6.py` into TuroArnis app
- [ ] Test end-to-end with real camera feed
- [ ] Viewpoint switching (front/left/right)

### Next Models
- [ ] Train right viewpoint specialist (v5 or v6)

## Known Limitations

1. **Right viewpoint pending** — right specialist model not yet trained
2. **Performance**: ~10-20 FPS on Intel Core 7 150U
   - **Solution**: Frame skipping or YOLO quantization
3. **CPU-Only**: No GPU acceleration
4. **Left viewpoint batch bugfix applied** — `front/` and `left/` model files include the fix, but original `src/` files do not
5. **Left-side classes weaker in front models**: crown 67%, left_chest 40%, left_elbow 7% (front viewpoint only)

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
