import torch
import torch.onnx
import os
import sys
import json
import numpy as np
import onnxruntime as ort
import joblib
import shutil
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from training.geopose.model import GeoPoseNet
from training.geopose.graph_builder import ArnisGraphBuilder

def export_model(model_path='models/v_geopose_best/model.pt', output_path='models/v_geopose_best/model.onnx'):
    device = torch.device('cpu')
    
    # Load metadata to get num_classes
    model_dir = os.path.dirname(model_path)
    meta_path = os.path.join(model_dir, 'metadata.json')
    num_classes = 13
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            num_classes = meta.get('num_classes', 13)
    
    print(f"Exporting model with {num_classes} classes...")
    
    # Load Model
    model = GeoPoseNet(num_classes=num_classes)
    # Check if model file contains state_dict or full model
    try:
        # Load logic slightly robustified
        payload = torch.load(model_path, map_location=device)
        if isinstance(payload, GeoPoseNet):
            model = payload
        elif isinstance(payload, dict):
            model.load_state_dict(payload)
        else:
            print("Unknown model format")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Create Dummy Input
    # 19 nodes (17 body + 2 weapon), 2 features (x,y normalized)
    num_nodes = 19
    dummy_x = torch.randn(num_nodes, 2)
    
    # Get Edge Index from GraphBuilder to ensure consistency
    builder = ArnisGraphBuilder()
    dummy_edge_index = builder.edge_index
    num_edges = dummy_edge_index.shape[1]
    
    # Dummy Edge Attributes (4 features: dist, sin, cos, rigid)
    dummy_edge_attr = torch.randn(num_edges, 4)
    
    # Dummy Batch (all 0 for single graph)
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long)
    
    input_names = ["x", "edge_index", "edge_attr", "batch"]
    output_names = ["logits"]
    
    dynamic_axes = {
        # Fixed size for MVP single-frame inference
    }
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch),
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=16, 
        dynamic_axes=dynamic_axes
    )
    
    print("ONNX export successful.")
    
    # Create artifacts
    print("Creating artifacts...")
    
    # 1. Scaler (Identity)
    scaler = StandardScaler()
    # Fit on dummy data just to valid attributes (mean=0, scale=1)
    scaler.fit(np.zeros((10, 2))) # 2 features
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print("- scaler.joblib created (Identity)")
    
    # 2. Label Encoder (Copy from data/processed)
    le_src = os.path.join(project_root, 'data', 'processed', 'label_encoder.joblib')
    if os.path.exists(le_src):
        shutil.copy(le_src, os.path.join(model_dir, 'label_encoder.joblib'))
        print("- label_encoder.joblib copied")
    else:
        print("Warning: label_encoder.joblib not found in data/processed")
        
    verify_onnx(output_path, dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch, model)

def verify_onnx(onnx_path, x, edge_index, edge_attr, batch, torch_model):
    print("Verifying ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # ONNX Runtime inputs
    ort_inputs = {
        "x": x.numpy(),
        "edge_index": edge_index.numpy(),
        "edge_attr": edge_attr.numpy(),
        "batch": batch.numpy()
    }
    
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Torch output
    with torch.no_grad():
        torch_out = torch_model(x, edge_index, edge_attr, batch)
    
    # Compare
    np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Verification Passed: PyTorch and ONNX outputs match.")

if __name__ == '__main__':
    export_model()
