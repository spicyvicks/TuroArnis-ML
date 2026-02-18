# Spatial GCN Workflow Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements_gcn.txt
```

### 2. Create Graph Dataset (Without Augmentation)
```bash
cd training
python create_graph_dataset.py --dataset_root ../dataset_split --output_root ../dataset_graphs
```

### 3. Create Graph Dataset (With 4x Augmentation)
```bash
python create_graph_dataset.py --dataset_root ../dataset_split --output_root ../dataset_graphs --augment
```

### 4. Train Model
```bash
python train_gcn.py --dataset_root ../dataset_graphs --max_epochs 200 --batch_size 32
```

### 5. Run Real-Time Inference
```bash
cd ..
python inference_realtime_gcn.py --model models/gcn_checkpoints/best_model.pth --camera 0
```

## Expected Results

### Dataset Creation
- **Without augmentation**: ~600 graphs
- **With augmentation**: ~2,400 graphs (4x multiplier)
- **Stick detection rate**: Target >80%

### Training
- **Target train accuracy**: 70-85% (not 97%!)
- **Target validation accuracy**: 60-75%
- **Train-val gap**: <15%
- **Training time**: ~30-60 minutes on GPU

### Inference
- **FPS**: >15 on GPU, >5 on CPU
- **Latency**: <100ms per frame
- **Stability**: Smooth predictions with deque buffer

## Troubleshooting

### PyTorch Geometric Installation Issues
If you encounter errors installing PyG, use:
```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Stick Detection Failures
If stick detection rate is <80%, check:
1. Stick detector model path is correct
2. Images have visible sticks
3. Consider using heuristic fallback (uncomment in `create_graph_dataset.py`)

### Low Accuracy
If validation accuracy <50%:
1. Try moderate augmentation (5x multiplier)
2. Increase hidden channels (64 → 128)
3. Train longer (200 → 300 epochs)
4. Collect more data

### Overfitting (High Train-Val Gap)
If train-val gap >20%:
1. Increase dropout (0.5 → 0.6)
2. Increase weight decay (1e-3 → 1e-2)
3. Reduce augmentation
4. Use early stopping (already implemented)
