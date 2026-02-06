
# TuroArnis GeoPose-Net (GNN)

This directory contains the Graph Neural Network implementation for TuroArnis.

## Components

1. **weapon_geometry.py**: 
   - Corrects "stick drift" using anatomical constraints (Alpha=1.35, Max Angle=20°).
   - Used during dataset creation and inference.

2. **graph_builder.py**:
   - Converts 19 keypoints (17 body + 2 stick) into a graph.
   - Edges include body skeleton + weapon connections (Wrist->Grip, Grip->Tip).

3. **create_dataset.py**:
   - Reads `dataset/` images.
   - Applies geometry correction.
   - Saves graphs to `data/processed/`.

4. **model.py**:
   - 2-Layer Graph Attention Network (GAT).
   - ~10k parameters.

5. **train.py**:
   - 5-Fold Cross Validation.
   - Saves models to `models/v_geopose_foldX`.

## Usage

**Train via Model Manager:**
1. Run `python training/model_manager.py`
2. Select "Train new model" -> "GeoPose GNN"

**Manual Training:**
```bash
python training/geopose/create_dataset.py
python training/geopose/train.py --epochs 100 --folds 5
```

## Troubleshooting Accuracy

If accuracy is low (~10-20%):
1. **Check Stick Detection**: Ensure YOLO is detecting sticks in the training data.
2. **Hyperparameters**: Increase `hidden_channels` in `model.py` (currently 64) or increase epochs.
3. **Data Norm**: Ensure inputs are normalized (currently 0-1 via image width/height).

## ONNX Export

Exporting GNNs to ONNX is experimental. If it fails, the app should be updated to load `.pt` files directly using `torch.load()`.
