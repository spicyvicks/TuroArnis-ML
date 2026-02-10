# Deployment Package Manifest
# Generated: 2026-02-10

## Package Information
- **Version**: 1.0
- **Target Platform**: Windows Desktop (CPU-only)
- **Python Version**: 3.11
- **PyTorch Version**: 2.10.0+cpu

## Files Included

### Models (Total: ~4.05 MB)
- models/hybrid_gcn_v2_front.pth (1.35 MB)
- models/hybrid_gcn_v2_left.pth (1.35 MB)
- models/hybrid_gcn_v2_right.pth (1.35 MB)

### Weights (Total: ~6.12 MB)
- weights/best.pt (6.12 MB) - YOLOv8-Pose stick detector

### Source Code
- src/model_architecture.py - HybridGCN model definition
- src/feature_extraction.py - Feature extraction utilities
- src/feature_templates.json (202 KB) - Reference pose templates

### Documentation
- docs/implementation_plan.md - Comprehensive implementation guide
- README.md - Quick start and usage guide

### Configuration
- requirements.txt - Python dependencies (locked versions)

## Total Package Size
Approximately **10.5 MB** (excluding Python environment)

## Deployment Checklist

### Pre-Implementation
- [ ] Review `docs/implementation_plan.md`
- [ ] Answer 4 critical questions in implementation plan
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies from `requirements.txt`

### Implementation Phase
- [ ] Create inference engine (`inference_engine.py`)
- [ ] Create model loader (`model_loader.py`)
- [ ] Create desktop app (`app_main.py` with PyQt6)
- [ ] Create camera handler (`camera_handler.py`)
- [ ] Create database layer (`database.py` with SQLite)
- [ ] Implement performance optimizations (quantization)

### Testing Phase
- [ ] Unit tests for feature extraction
- [ ] Integration tests for end-to-end pipeline
- [ ] Performance benchmarking (target: 30 FPS)
- [ ] Real-time classification testing
- [ ] Viewpoint switching testing
- [ ] Database logging verification

### Packaging Phase
- [ ] Create PyInstaller spec file
- [ ] Build standalone executable
- [ ] Test on clean Windows machine
- [ ] Verify .exe size < 600MB
- [ ] Verify startup time < 10 seconds

## Known Limitations

1. **Performance**: Current pipeline runs at ~10 FPS on Intel Core 7 150U
   - **Solution**: Implement YOLO quantization or frame skipping (see implementation plan)

2. **Stick Detection Dependency**: Models require YOLO stick detector
   - **Fallback**: Can use default stick positions if detection fails

3. **CPU-Only**: No GPU acceleration
   - **Note**: Models are optimized for CPU inference

4. **Viewpoint Selection**: Requires manual selection or auto-detection implementation
   - **Decision needed**: See implementation plan questions

## Support Files Required (Not Included)

These files are needed for full implementation but not included in this package:

1. **Webcam drivers**: System-dependent
2. **SQLite database**: Created at runtime
3. **Application icon**: For .exe packaging
4. **User manual**: For end users

## Version History

### v1.0 (2026-02-10)
- Initial deployment package
- 3 specialist models (Front/Left/Right)
- YOLO stick detector
- Feature extraction utilities
- Comprehensive documentation

## Next Release Plans

- [ ] Quantized models (INT8) for faster inference
- [ ] ONNX export for cross-platform compatibility
- [ ] Auto-viewpoint detection model
- [ ] Mobile deployment (TensorFlow Lite)

### v1.1 (2026-02-11)
- Updated Class List (Removed 'neutral_stance', 13 -> 12 classes)
- Syncing latest Hybrid GCN V2 models
- Updated feature extraction logic
