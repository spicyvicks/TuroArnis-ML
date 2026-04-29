# Deployment Package Manifest
# Updated: 2026-04-28

## Package Information
- **Version**: v5_front
- **Target Platform**: Windows Desktop (CPU-only)
- **Python Version**: 3.11
- **PyTorch Version**: 2.10.0+cpu

## Active Model

**`model_front_v5_deploy.pth`** — Front-view specialist, 77.7% real-only accuracy
- 13 classes (including `neutral`)
- 6-dim node features + 46 hybrid features
- 69,541 parameters
- CPU inference optimized

## Files Included

### Models
- `models/model_front_v5_deploy.pth` (861 KB) — Active deployment model
- Legacy: `models/hybrid_gcn_v2_front.pth` (1.35 MB) — Superseded by v5

### Weights
- `weights/best.pt` (6.12 MB) — YOLOv8-Pose stick detector

### Source Code
- `src/model_v5.py` — V5 HybridGCN architecture (exact training match)
- `src/feature_extraction_v5.py` — V5 feature extraction with signed features
- `src/inference_v5.py` — High-level inference wrapper for app integration
- `src/model_v6.py` — V6 HybridGCN with masked pooling + node_mask
- `src/feature_extraction_v6.py` — V6 feature extraction with 7-dim nodes + node_mask
- `src/inference_v6.py` — V6 high-level inference wrapper
- `src/model_architecture.py` — Legacy architecture (superseded)
- `src/feature_extraction.py` — Legacy extraction (superseded)
- `src/feature_templates.json` (202 KB) — Reference pose templates

### Documentation
- `README_v5.md` — V5 deployment guide (**RECOMMENDED for production**)
- `README_v6.md` — V6 deployment guide (masked pooling architecture)
- `docs/implementation_plan.md` — Legacy comprehensive guide
- `README.md` — Legacy quick start

### Configuration
- `requirements.txt` — Python dependencies (locked versions)

## Total Package Size
Approximately **8.5 MB** (excluding Python environment)

## Deployment Checklist

### Pre-Implementation
- [x] Model trained and validated (77.7% real-only)
- [x] Deployment checkpoint created with config
- [x] Feature extraction verified against training
- [ ] **App update: Add `neutral` class to class list (13 classes)**
- [ ] **App update: Implement signed direction features in feature extraction**
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies from `requirements.txt`

### Implementation Phase
- [ ] Integrate `inference_v5.py` into TuroArnis app
- [ ] Test end-to-end with real camera feed
- [ ] Viewpoint switching (currently front-only)

### Next Models
- [ ] Train left viewpoint specialist (v5 or v2)
- [ ] Train right viewpoint specialist (v5 or v2)

## Known Limitations

1. **Front-view only** — left/right models pending
2. **Performance**: ~10-20 FPS on Intel Core 7 150U
   - **Solution**: Frame skipping or YOLO quantization
3. **CPU-Only**: No GPU acceleration
4. **Left-side classes weaker**: crown 73%, left_chest 33%, left_elbow 20%

## Version History

### v6_front (2026-04-28)
- **V6 front specialist model deployed**
- 74.6% real-only test accuracy (masked pooling architecture)
- 13 classes with `neutral` support
- 7-dim node features (has_stick binary on nodes)
- 49 hybrid features (signed + has_stick)
- True zero-stick fallback with node_mask masking
- Cleaner architecture, 3.1% below v5 but more robust

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
